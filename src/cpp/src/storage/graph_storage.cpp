//
// Created by Jason Mohoney on 6/18/21.
//

#include "storage/graph_storage.h"

#include <algorithm>
#include <random>

#include "data/ordering.h"
#include "reporting/logger.h"
#include <torch/script.h>
#include <set>
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;

torch::Tensor make_unique(torch::Tensor input) {
    input = input.to(torch::kCPU);
    int64_t* data = input.data_ptr<int64_t>();
    int64_t size = input.size(0);

    std::set<int64_t> unique_set(data, data + size);
    std::vector<int64_t> unique_vec(unique_set.begin(), unique_set.end());

    torch::Tensor result = torch::from_blob(unique_vec.data(), {static_cast<int64_t>(unique_vec.size())}, torch::kInt64).clone();
    return result;
}


GraphModelStorage::GraphModelStorage(GraphModelStoragePtrs storage_ptrs, shared_ptr<StorageConfig> storage_config) {
    storage_ptrs_ = storage_ptrs;
    train_ = true;
    full_graph_evaluation_ = storage_config->full_graph_evaluation;

    prefetch_ = storage_config->prefetch;
    prefetch_complete_ = false;
    subgraph_lock_ = new std::mutex();
    subgraph_cv_ = new std::condition_variable();

    current_subgraph_state_ = nullptr;
    next_subgraph_state_ = nullptr;
    in_memory_embeddings_ = nullptr;
    in_memory_features_ = nullptr;

    num_nodes_ = storage_config->dataset->num_nodes;
    num_edges_ = storage_config->dataset->num_edges;

    if (storage_ptrs_.node_features != nullptr &&
        instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features)) {

        string node_feature_filename =
            storage_config->dataset->dataset_dir +
            PathConstants::nodes_directory +
            PathConstants::features_file +
            PathConstants::file_ext;

        SPDLOG_INFO(">>> Forcing in_memory_features_ init: {}", node_feature_filename);

        in_memory_features_ = std::make_shared<InMemory>(
            node_feature_filename,
            storage_ptrs_.node_features->dim0_size_,
            storage_ptrs_.node_features->dim1_size_,
            storage_ptrs_.node_features->dtype_,
            torch::kCPU
        );

        storage_ptrs_.node_features = in_memory_features_;

        SPDLOG_INFO(">>> Replaced storage_ptrs_.node_features with in_memory_features_");
    }

}

GraphModelStorage::GraphModelStorage(GraphModelStoragePtrs storage_ptrs, bool prefetch) {
    storage_ptrs_ = storage_ptrs;
    train_ = true;
    full_graph_evaluation_ = false;

    prefetch_ = prefetch;
    prefetch_complete_ = false;
    subgraph_lock_ = new std::mutex();
    subgraph_cv_ = new std::condition_variable();

    current_subgraph_state_ = nullptr;
    next_subgraph_state_ = nullptr;
    in_memory_embeddings_ = nullptr;
    in_memory_features_ = nullptr;

    if (storage_ptrs_.node_embeddings != nullptr) {
        num_nodes_ = storage_ptrs_.node_embeddings->getDim0();
    } else if (storage_ptrs_.node_features != nullptr) {
        num_nodes_ = storage_ptrs_.node_features->getDim0();
    } else {
        throw MariusRuntimeException("The input graph must have node features and/or node embeddings");
    }
    num_edges_ = storage_ptrs_.edges->getDim0();
}

GraphModelStorage::~GraphModelStorage() {
    unload(false);

    delete subgraph_lock_;
    delete subgraph_cv_;
}

void GraphModelStorage::_load(shared_ptr<Storage> storage) {
    if (storage != nullptr) {
        storage->load();
    }
}

void GraphModelStorage::_unload(shared_ptr<Storage> storage, bool write) {
    if (storage != nullptr) {
        storage->unload(write);
    }
}

void GraphModelStorage::load() {
    _load(storage_ptrs_.edges);
    _load(storage_ptrs_.train_edges);
    _load(storage_ptrs_.train_edges_dst_sort);
    _load(storage_ptrs_.nodes);

    if (train_) {
        _load(storage_ptrs_.node_embeddings);
        _load(storage_ptrs_.node_optimizer_state);
        _load(storage_ptrs_.node_features);
    } else {
        if (storage_ptrs_.node_embeddings != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings) && full_graph_evaluation_) {
                _load(in_memory_embeddings_);
            } else {
                _load(storage_ptrs_.node_embeddings);
            }
        }

        if (storage_ptrs_.node_features != nullptr) {
            if (instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_features) && full_graph_evaluation_) {
                _load(in_memory_features_);
            } else {
                _load(storage_ptrs_.node_features);
            }
        }
    }

    _load(storage_ptrs_.encoded_nodes);

    _load(storage_ptrs_.node_labels);
    _load(storage_ptrs_.relation_features);
    if (train_ && !cached_critical_features_.defined()) {
        std::string critical_nodes_path = "datasets/ogbn_arxiv/nodes/critical_nodes.bin";
        std::ifstream infile(critical_nodes_path, std::ios::binary | std::ios::ate);

        if (infile.is_open()) {
            std::streamsize size = infile.tellg();
            infile.seekg(0, std::ios::beg);

            int64_t num_elems = size / sizeof(int64_t);
            std::vector<int64_t> buffer(num_elems);
            if (infile.read(reinterpret_cast<char*>(buffer.data()), size)) {
                torch::Tensor critical_nodes = torch::from_blob(buffer.data(), {num_elems}, torch::kInt64).clone();
                torch::Tensor filtered = critical_nodes.masked_select((critical_nodes >= 0) & (critical_nodes < num_nodes_));
                critical_nodes = make_unique(filtered);
                critical_node_ids_ = critical_nodes.to(torch::kCUDA);

                SPDLOG_INFO("After filtering, valid critical_nodes count: {}", critical_nodes.size(0));
                SPDLOG_INFO("critical_nodes max: {}, total_nodes: {}", critical_nodes.max().item<int64_t>(), num_nodes_);

                SPDLOG_INFO(">> Trying to fetch features...");
                torch::Tensor feats = getNodeFeatures(critical_nodes);
                SPDLOG_INFO(">> Got features, shape: [{}]", fmt::join(feats.sizes(), ", "));

                if (feats.defined() && feats.numel() > 0) {
                    cached_critical_features_ = feats.to(torch::kCUDA);

                    critical_id_to_index_ = torch::full({num_nodes_}, -1, torch::kInt64).to(torch::kCUDA);
                    critical_id_to_index_.index_put_({critical_node_ids_}, torch::arange(critical_node_ids_.size(0), torch::kInt64).to(torch::kCUDA));

                    SPDLOG_INFO(">> Critical features moved to GPU.");
                    SPDLOG_INFO(">> cached_critical_features_ on device: {}", cached_critical_features_.device().str());

                    auto sizes_vec = std::vector<int64_t>(cached_critical_features_.sizes().begin(), cached_critical_features_.sizes().end());
                    SPDLOG_INFO(">> cached_critical_features_ shape: [{}]", fmt::join(sizes_vec, ", "));
                } else {
                    SPDLOG_ERROR(">> Fetched features are empty! Aborting GPU move.");
                }

            } else {
                SPDLOG_ERROR("Failed to read content from {}", critical_nodes_path);
            }
            infile.close();
        } else {
            SPDLOG_WARN("Could not open critical_nodes.bin at {}", critical_nodes_path);
        }
    }

}

void GraphModelStorage::unload(bool write) {
    _unload(storage_ptrs_.edges, false);
    _unload(storage_ptrs_.train_edges, false);
    _unload(storage_ptrs_.train_edges_dst_sort, false);
    _unload(storage_ptrs_.validation_edges, false);
    _unload(storage_ptrs_.test_edges, false);
    _unload(storage_ptrs_.nodes, false);
    _unload(storage_ptrs_.train_nodes, false);
    _unload(storage_ptrs_.valid_nodes, false);
    _unload(storage_ptrs_.test_nodes, false);
    _unload(storage_ptrs_.node_embeddings, write);
    _unload(storage_ptrs_.encoded_nodes, write);
    _unload(storage_ptrs_.node_optimizer_state, write);
    _unload(storage_ptrs_.node_features, false);
    _unload(storage_ptrs_.relation_features, false);

    _unload(in_memory_embeddings_, false);
    _unload(in_memory_features_, false);

    for (auto f_edges : storage_ptrs_.filter_edges) {
        _unload(f_edges, false);
    }

    active_edges_ = torch::Tensor();
    active_nodes_ = torch::Tensor();
}

void GraphModelStorage::setEdgesStorage(shared_ptr<Storage> edge_storage) { storage_ptrs_.edges = edge_storage; }

void GraphModelStorage::setNodesStorage(shared_ptr<Storage> node_storage) { storage_ptrs_.nodes = node_storage; }

EdgeList GraphModelStorage::getEdges(Indices indices) {
    if (active_edges_.defined()) {
        return active_edges_.index_select(0, indices);
    } else {
        return storage_ptrs_.edges->indexRead(indices);
    }
}

EdgeList GraphModelStorage::getEdgesRange(int64_t start, int64_t size) {
    if (active_edges_.defined()) {
        return active_edges_.narrow(0, start, size);
    } else {
        return storage_ptrs_.edges->range(start, size);
    }
}

void GraphModelStorage::shuffleEdges() { storage_ptrs_.edges->shuffle(); }

Indices GraphModelStorage::getRandomNodeIds(int64_t size) {
    torch::TensorOptions ind_opts = torch::TensorOptions().dtype(torch::kInt64).device(storage_ptrs_.edges->device_);

    Indices ret;
    if (useInMemorySubGraph()) {
        if (storage_ptrs_.node_embeddings != nullptr) {
            ret = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getRandomIds(size);
        } else {
            ret = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getRandomIds(size);
        }
    } else {
        ret = torch::randint(getNumNodesInMemory(), {size}, ind_opts);
    }

    return ret;
}

Indices GraphModelStorage::getNodeIdsRange(int64_t start, int64_t size) {
    if (active_nodes_.defined()) {
        return active_nodes_.narrow(0, start, size);
    } else {
        return storage_ptrs_.nodes->range(start, size).flatten(0, 1);
    }
}

torch::Tensor GraphModelStorage::getNodeEmbeddings(Indices indices) {
    if (!train_ && instance_of<Storage, PartitionBufferStorage>(storage_ptrs_.node_embeddings) && full_graph_evaluation_) {
        if (in_memory_embeddings_ != nullptr) {
            return in_memory_embeddings_->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    } else {
        if (storage_ptrs_.node_embeddings != nullptr) {
            return storage_ptrs_.node_embeddings->indexRead(indices);
        } else {
            return torch::Tensor();
        }
    }
}

torch::Tensor GraphModelStorage::getNodeEmbeddingsRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_embeddings != nullptr) {
        return storage_ptrs_.node_embeddings->range(start, size);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getEncodedNodes(Indices indices) {
    if (storage_ptrs_.encoded_nodes != nullptr) {
        return storage_ptrs_.encoded_nodes->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getEncodedNodesRange(int64_t start, int64_t size) {
    if (storage_ptrs_.encoded_nodes != nullptr) {
        return storage_ptrs_.encoded_nodes->range(start, size);
    } else {
        return torch::Tensor();
    }
}


torch::Tensor GraphModelStorage::getNodeFeatures(torch::Tensor indices) {
    using namespace std::chrono;

    int64_t feature_row_count = 0;
    if (in_memory_features_ != nullptr) {
        feature_row_count = in_memory_features_->getDim0();
    } else if (storage_ptrs_.node_features != nullptr) {
        feature_row_count = storage_ptrs_.node_features->getDim0();
    }

    indices = indices.masked_select(indices < feature_row_count);
    if (indices.numel() == 0) {
        return torch::empty({0, storage_ptrs_.node_features->dim1_size_}, torch::kFloat32);
    }

    torch::Device dev_gpu = torch::kCUDA;
    torch::Device dev_cpu = torch::kCPU;

    if (cached_critical_features_.defined() && critical_id_to_index_.defined()) {
        auto t_total_start = high_resolution_clock::now();

        indices = indices.to(dev_gpu);
        int64_t max_idx = indices.max().item<int64_t>();
        int64_t map_size = critical_id_to_index_.size(0);
        if (max_idx >= map_size) {
            SPDLOG_ERROR("[CriticalMap] Index {} out of range for critical_id_to_index_ of size {}", max_idx, map_size);
            throw std::out_of_range("critical_id_to_index_ index out of range");
        }
        auto index_map = critical_id_to_index_.index_select(0, indices);
        auto is_critical = index_map >= 0;

        auto critical_batch_idx = torch::nonzero(is_critical).squeeze();
        auto non_critical_batch_idx = torch::nonzero(~is_critical).squeeze();

        torch::Tensor emb_critical, emb_non_critical;
        int64_t feat_dim = cached_critical_features_.size(1);
        int64_t total_size = indices.size(0);

        if (critical_batch_idx.numel() > 0) {
            auto t1 = high_resolution_clock::now();
            auto cache_indices = index_map.masked_select(is_critical);
            emb_critical = cached_critical_features_.index_select(0, cache_indices);
            auto t2 = high_resolution_clock::now();
            // SPDLOG_INFO("[Timer] GPU cached feature fetch: {:.3f} ms", duration<double, std::milli>(t2 - t1).count());
        } else {
            emb_critical = torch::empty({0, feat_dim}, torch::TensorOptions().device(dev_gpu));
        }

        if (non_critical_batch_idx.numel() > 0) {
            auto t3 = high_resolution_clock::now();
            auto ids = indices.index_select(0, non_critical_batch_idx).to(dev_cpu, true);

            if (in_memory_features_ != nullptr) {
                emb_non_critical = in_memory_features_->indexRead(ids);
            } else if (storage_ptrs_.node_features != nullptr) {
                emb_non_critical = storage_ptrs_.node_features->indexRead(ids);
            } else {
                SPDLOG_ERROR("No node_features available.");
                return torch::Tensor();
            }
            auto t4 = high_resolution_clock::now();
            // SPDLOG_INFO("[Timer] CPU feature fetch: {:.3f} ms", duration<double, std::milli>(t4 - t3).count());

            auto t5 = high_resolution_clock::now();
            if (emb_non_critical.is_contiguous()) {
                emb_non_critical = emb_non_critical.pin_memory();
            }
            emb_non_critical = emb_non_critical.to(dev_gpu, true);
            auto t6 = high_resolution_clock::now();
            // SPDLOG_INFO("[Timer] CPU→GPU transfer: {:.3f} ms", duration<double, std::milli>(t6 - t5).count());
        } else {
            emb_non_critical = torch::empty({0, feat_dim}, torch::TensorOptions().device(dev_gpu));
        }

        auto full_features = torch::empty({total_size, feat_dim}, emb_critical.options());
        if (critical_batch_idx.numel() > 0)
            full_features.index_copy_(0, critical_batch_idx, emb_critical);
        if (non_critical_batch_idx.numel() > 0)
            full_features.index_copy_(0, non_critical_batch_idx, emb_non_critical);

        auto t_total_end = high_resolution_clock::now();
        // SPDLOG_INFO("Total feature fetch time: {:.3f} ms", duration<double, std::milli>(t_total_end - t_total_start).count());

        return full_features;
    }

    // fallback with safety check
    indices = indices.to(dev_cpu);

    // Print fallback path status
    SPDLOG_INFO("[Fallback] Entered fallback path for node feature fetch");
    if (storage_ptrs_.node_features == nullptr) {
        SPDLOG_ERROR("[Fallback] node_features storage is nullptr!");
        throw std::runtime_error("Fallback node_features is nullptr");
    }
    if (!storage_ptrs_.node_features->isInitialized()) {
        SPDLOG_ERROR("[Fallback] node_features storage is not initialized!");
        throw std::runtime_error("Fallback node_features is not initialized");
    }

    int64_t max_id = indices.max().item<int64_t>();
    int64_t num_rows = storage_ptrs_.node_features->getDim0();
    int64_t feat_dim = storage_ptrs_.node_features->dim1_size_;

    if (max_id >= num_rows) {
        SPDLOG_ERROR("[Fallback] Node ID {} out of range (max valid = {})", max_id, num_rows - 1);
        throw std::out_of_range("Node ID exceeds available fallback rows");
    }

    SPDLOG_INFO("[Fallback] Fallback node_features_ shape: [{} x {}]", num_rows, feat_dim);

    return storage_ptrs_.node_features->indexRead(indices);

}


torch::Tensor GraphModelStorage::getNodeFeaturesRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_features != nullptr) {
        return storage_ptrs_.node_features->range(start, size);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getNodeLabels(Indices indices) {
    if (storage_ptrs_.node_labels != nullptr) {
        return storage_ptrs_.node_labels->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

torch::Tensor GraphModelStorage::getNodeLabelsRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_labels != nullptr) {
        return storage_ptrs_.node_labels->range(start, size);
    } else {
        return torch::Tensor();
    }
}

void GraphModelStorage::updatePutNodeEmbeddings(Indices indices, torch::Tensor embeddings) { storage_ptrs_.node_embeddings->indexPut(indices, embeddings); }

void GraphModelStorage::updateAddNodeEmbeddings(Indices indices, torch::Tensor values) { storage_ptrs_.node_embeddings->indexAdd(indices, values); }

void GraphModelStorage::updatePutEncodedNodes(Indices indices, torch::Tensor values) { storage_ptrs_.encoded_nodes->indexPut(indices, values); }

void GraphModelStorage::updatePutEncodedNodesRange(int64_t start, int64_t size, torch::Tensor values) {
    storage_ptrs_.encoded_nodes->rangePut(start, size, values);
}

OptimizerState GraphModelStorage::getNodeEmbeddingState(Indices indices) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        return storage_ptrs_.node_optimizer_state->indexRead(indices);
    } else {
        return torch::Tensor();
    }
}

OptimizerState GraphModelStorage::getNodeEmbeddingStateRange(int64_t start, int64_t size) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        return storage_ptrs_.node_optimizer_state->range(start, size);
    } else {
        return torch::Tensor();
    }
}

void GraphModelStorage::updatePutNodeEmbeddingState(Indices indices, OptimizerState state) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        storage_ptrs_.node_optimizer_state->indexPut(indices, state);
    }
}

void GraphModelStorage::updateAddNodeEmbeddingState(Indices indices, torch::Tensor values) {
    if (storage_ptrs_.node_optimizer_state != nullptr) {
        storage_ptrs_.node_optimizer_state->indexAdd(indices, values);
    }
}

bool GraphModelStorage::embeddingsOffDevice() {
    if (storage_ptrs_.node_embeddings != nullptr) {
        return storage_ptrs_.node_embeddings->device_ != torch::kCUDA;
    } else if (storage_ptrs_.node_features != nullptr) {
        return storage_ptrs_.node_features->device_ != torch::kCUDA;
    } else {
        return false;
    }
}

void GraphModelStorage::initializeInMemorySubGraph(torch::Tensor buffer_state, int num_hash_maps) {
    if (useInMemorySubGraph()) {
        current_subgraph_state_ = std::make_shared<InMemorySubgraphState>();

        buffer_state = buffer_state.to(torch::kInt64);

        int buffer_size = buffer_state.size(0);
        int num_edge_buckets_in_mem = buffer_size * buffer_size;
        int num_partitions = getNumPartitions();

        torch::Tensor new_in_mem_partition_ids = buffer_state;
        auto new_in_mem_partition_ids_accessor = new_in_mem_partition_ids.accessor<int64_t, 1>();

        torch::Tensor in_mem_edge_bucket_ids = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
        torch::Tensor in_mem_edge_bucket_sizes = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
        torch::Tensor global_edge_bucket_starts = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);

        auto in_mem_edge_bucket_ids_accessor = in_mem_edge_bucket_ids.accessor<int64_t, 1>();
        auto in_mem_edge_bucket_sizes_accessor = in_mem_edge_bucket_sizes.accessor<int64_t, 1>();
        auto global_edge_bucket_starts_accessor = global_edge_bucket_starts.accessor<int64_t, 1>();

        // TODO we don't need to do this every time
        std::vector<int64_t> edge_bucket_sizes_ = storage_ptrs_.edges->getEdgeBucketSizes();
        torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int)edge_bucket_sizes_.size()}, torch::kInt64);
        torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
        torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;
        auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
        auto edge_bucket_starts_disk_accessor = edge_bucket_starts_disk.accessor<int64_t, 1>();

#pragma omp parallel for
        for (int i = 0; i < buffer_size; i++) {
            for (int j = 0; j < buffer_size; j++) {
                int64_t edge_bucket_id = new_in_mem_partition_ids_accessor[i] * num_partitions + new_in_mem_partition_ids_accessor[j];
                int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
                int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

                int idx = i * buffer_size + j;
                in_mem_edge_bucket_ids_accessor[idx] = edge_bucket_id;
                in_mem_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
                global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
            }
        }

        torch::Tensor in_mem_edge_bucket_starts = in_mem_edge_bucket_sizes.cumsum(0);
        int64_t total_size = in_mem_edge_bucket_starts[-1].item<int64_t>();
        in_mem_edge_bucket_starts = in_mem_edge_bucket_starts - in_mem_edge_bucket_sizes;

        auto in_mem_edge_bucket_starts_accessor = in_mem_edge_bucket_starts.accessor<int64_t, 1>();

        current_subgraph_state_->all_in_memory_edges_ = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);

#pragma omp parallel for
        for (int i = 0; i < num_edge_buckets_in_mem; i++) {
            int64_t edge_bucket_size = in_mem_edge_bucket_sizes_accessor[i];
            int64_t edge_bucket_start = global_edge_bucket_starts_accessor[i];
            int64_t local_offset = in_mem_edge_bucket_starts_accessor[i];

            current_subgraph_state_->all_in_memory_edges_.narrow(0, local_offset, edge_bucket_size) =
                storage_ptrs_.edges->range(edge_bucket_start, edge_bucket_size);
        }

        if (storage_ptrs_.node_embeddings != nullptr) {
            current_subgraph_state_->global_to_local_index_map_ =
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getGlobalToLocalMap(true);
        } else if (storage_ptrs_.node_features != nullptr) {
            current_subgraph_state_->global_to_local_index_map_ =
                std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getGlobalToLocalMap(true);
        }

        torch::Tensor mapped_edges;
        torch::Tensor mapped_edges_dst_sort;
        if (storage_ptrs_.edges->dim1_size_ == 3) {
            mapped_edges =
                torch::stack({current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, 0)),
                              current_subgraph_state_->all_in_memory_edges_.select(1, 1),
                              current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, -1))})
                    .transpose(0, 1);
        } else if (storage_ptrs_.edges->dim1_size_ == 2) {
            mapped_edges =
                torch::stack({current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, 0)),
                              current_subgraph_state_->global_to_local_index_map_.index_select(0, current_subgraph_state_->all_in_memory_edges_.select(1, -1))})
                    .transpose(0, 1);
        } else {
            // TODO use a function for logging errors and throwing expections
            SPDLOG_ERROR("Unexpected number of edge columns");
            std::runtime_error("Unexpected number of edge columns");
        }

        current_subgraph_state_->all_in_memory_mapped_edges_ = mapped_edges;

        mapped_edges = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, true);
        mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, false);

        mapped_edges = mapped_edges.to(torch::kInt64);
        mapped_edges_dst_sort = mapped_edges_dst_sort.to(torch::kInt64);

        if (current_subgraph_state_->in_memory_subgraph_ != nullptr) {
            current_subgraph_state_->in_memory_subgraph_ = nullptr;
        }

        current_subgraph_state_->in_memory_subgraph_ = std::make_shared<MariusGraph>(mapped_edges, mapped_edges_dst_sort, getNumNodesInMemory(), num_hash_maps);

        current_subgraph_state_->in_memory_partition_ids_ = new_in_mem_partition_ids;
        current_subgraph_state_->in_memory_edge_bucket_ids_ = in_mem_edge_bucket_ids;
        current_subgraph_state_->in_memory_edge_bucket_sizes_ = in_mem_edge_bucket_sizes;
        current_subgraph_state_->in_memory_edge_bucket_starts_ = in_mem_edge_bucket_starts;

        if (prefetch_) {
            if (hasSwap()) {
                // update next_subgraph_state_ in background
                getNextSubGraph();
            }
        }
    } else {
        // Either nothing buffered (in memory training) or eval and doing full graph evaluation
        current_subgraph_state_ = std::make_shared<InMemorySubgraphState>();

        bool should_sort = false;

        EdgeList src_sort;
        EdgeList dst_sort;
        if (storage_ptrs_.train_edges != nullptr) {
            src_sort = storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()).to(torch::kInt64);
            if (storage_ptrs_.train_edges_dst_sort != nullptr) {
                dst_sort = storage_ptrs_.train_edges_dst_sort->range(0, storage_ptrs_.train_edges_dst_sort->getDim0()).to(torch::kInt64);
            } else {
                dst_sort = storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()).to(torch::kInt64);
                should_sort = true;
            }
        } else {
            src_sort = storage_ptrs_.edges->range(0, storage_ptrs_.edges->getDim0()).to(torch::kInt64);
            dst_sort = storage_ptrs_.edges->range(0, storage_ptrs_.edges->getDim0()).to(torch::kInt64);
            should_sort = true;
        }

        if (should_sort) {
            src_sort = src_sort.index_select(0, torch::argsort(src_sort.select(1, 0))).to(torch::kInt64);
            dst_sort = dst_sort.index_select(0, torch::argsort(dst_sort.select(1, -1))).to(torch::kInt64);
        }

        current_subgraph_state_->in_memory_subgraph_ = std::make_shared<MariusGraph>(src_sort, dst_sort, getNumNodesInMemory(), num_hash_maps);
    }
}

void GraphModelStorage::updateInMemorySubGraph() {
    if (prefetch_) {
        // wait until the prefetching has been completed
        std::unique_lock lock(*subgraph_lock_);
        subgraph_cv_->wait(lock, [this] { return prefetch_complete_ == true; });
        // need to wait for the subgraph to be prefetched to perform the swap, otherwise the prefetched buffer_index_map may be incorrect
        performSwap();
        // free previous subgraph
        current_subgraph_state_->in_memory_subgraph_ = nullptr;
        current_subgraph_state_ = nullptr;

        current_subgraph_state_ = next_subgraph_state_;
        next_subgraph_state_ = nullptr;
        prefetch_complete_ = false;

        if (hasSwap()) {
            // update next_subgraph_state_ in background
            getNextSubGraph();
        }
    } else {
        std::pair<std::vector<int>, std::vector<int>> current_swap_ids = getNextSwapIds();
        performSwap();
        updateInMemorySubGraph_(current_subgraph_state_, current_swap_ids);
    }
}

void GraphModelStorage::getNextSubGraph() {
    std::pair<std::vector<int>, std::vector<int>> next_swap_ids = getNextSwapIds();
    next_subgraph_state_ = std::make_shared<InMemorySubgraphState>();
    next_subgraph_state_->in_memory_subgraph_ = nullptr;
    std::thread(&GraphModelStorage::updateInMemorySubGraph_, this, next_subgraph_state_, next_swap_ids).detach();
}

void GraphModelStorage::updateInMemorySubGraph_(shared_ptr<InMemorySubgraphState> subgraph, std::pair<std::vector<int>, std::vector<int>> swap_ids) {
    if (prefetch_) {
        subgraph_lock_->lock();
    }

    std::vector<int> evict_partition_ids = std::get<0>(swap_ids);
    std::vector<int> admit_partition_ids = std::get<1>(swap_ids);

    torch::Tensor admit_ids_tensor = torch::tensor(admit_partition_ids, torch::kCPU);

    int buffer_size = current_subgraph_state_->in_memory_partition_ids_.size(0);
    int num_edge_buckets_in_mem = current_subgraph_state_->in_memory_edge_bucket_ids_.size(0);
    int num_partitions = getNumPartitions();
    int num_swap_partitions = evict_partition_ids.size();
    int num_remaining_partitions = buffer_size - num_swap_partitions;

    // get edge buckets that will be kept in memory
    torch::Tensor keep_mask = torch::ones({num_edge_buckets_in_mem}, torch::kBool);
    auto accessor_keep_mask = keep_mask.accessor<bool, 1>();
    auto accessor_in_memory_edge_bucket_ids_ = current_subgraph_state_->in_memory_edge_bucket_ids_.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < num_edge_buckets_in_mem; i++) {
        int64_t edge_bucket_id = accessor_in_memory_edge_bucket_ids_[i];
        int64_t src_partition = edge_bucket_id / num_partitions;
        int64_t dst_partition = edge_bucket_id % num_partitions;

        for (int j = 0; j < num_swap_partitions; j++) {
            if (src_partition == evict_partition_ids[j] || dst_partition == evict_partition_ids[j]) {
                accessor_keep_mask[i] = false;
            }
        }
    }

    torch::Tensor in_mem_edge_bucket_ids = current_subgraph_state_->in_memory_edge_bucket_ids_.masked_select(keep_mask);
    torch::Tensor in_mem_edge_bucket_sizes = current_subgraph_state_->in_memory_edge_bucket_sizes_.masked_select(keep_mask);
    torch::Tensor local_or_global_edge_bucket_starts = current_subgraph_state_->in_memory_edge_bucket_starts_.masked_select(keep_mask);

    // get new in memory partition ids
    keep_mask = torch::ones({buffer_size}, torch::kBool);
    accessor_keep_mask = keep_mask.accessor<bool, 1>();
    auto accessor_in_memory_partition_ids_ = current_subgraph_state_->in_memory_partition_ids_.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        int64_t partition_id = accessor_in_memory_partition_ids_[i];

        for (int j = 0; j < num_swap_partitions; j++) {
            if (partition_id == evict_partition_ids[j]) {
                accessor_keep_mask[i] = false;
                break;
            }
        }
    }

    torch::Tensor old_in_mem_partition_ids = current_subgraph_state_->in_memory_partition_ids_.masked_select(keep_mask);
    torch::Tensor new_in_mem_partition_ids = current_subgraph_state_->in_memory_partition_ids_.masked_scatter(~keep_mask, admit_ids_tensor);
    auto old_in_mem_partition_ids_accessor = old_in_mem_partition_ids.accessor<int64_t, 1>();
    auto new_in_mem_partition_ids_accessor = new_in_mem_partition_ids.accessor<int64_t, 1>();

    // get new incoming edge buckets
    int num_new_edge_buckets = num_swap_partitions * (num_remaining_partitions + buffer_size);

    torch::Tensor new_edge_bucket_ids = torch::zeros({num_new_edge_buckets}, torch::kInt64);
    torch::Tensor new_edge_bucket_sizes = torch::zeros({num_new_edge_buckets}, torch::kInt64);
    torch::Tensor new_global_edge_bucket_starts = torch::zeros({num_new_edge_buckets}, torch::kInt64);

    auto new_edge_bucket_ids_accessor = new_edge_bucket_ids.accessor<int64_t, 1>();
    auto new_edge_bucket_sizes_accessor = new_edge_bucket_sizes.accessor<int64_t, 1>();
    auto new_global_edge_bucket_starts_accessor = new_global_edge_bucket_starts.accessor<int64_t, 1>();

    // TODO we don't need to do this every time
    std::vector<int64_t> edge_bucket_sizes_ = storage_ptrs_.edges->getEdgeBucketSizes();
    torch::Tensor edge_bucket_sizes = torch::from_blob(edge_bucket_sizes_.data(), {(int)edge_bucket_sizes_.size()}, torch::kInt64);
    torch::Tensor edge_bucket_ends_disk = edge_bucket_sizes.cumsum(0);
    torch::Tensor edge_bucket_starts_disk = edge_bucket_ends_disk - edge_bucket_sizes;
    auto edge_bucket_sizes_accessor = edge_bucket_sizes.accessor<int64_t, 1>();
    auto edge_bucket_starts_disk_accessor = edge_bucket_starts_disk.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < num_remaining_partitions; i++) {
        for (int j = 0; j < num_swap_partitions; j++) {
            int64_t edge_bucket_id = old_in_mem_partition_ids_accessor[i] * num_partitions + admit_partition_ids[j];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
            int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

            int idx = i * num_swap_partitions + j;
            new_edge_bucket_ids_accessor[idx] = edge_bucket_id;
            new_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
            new_global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
        }
    }

    int offset = num_swap_partitions * num_remaining_partitions;

#pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < num_swap_partitions; j++) {
            int64_t edge_bucket_id = admit_partition_ids[j] * num_partitions + new_in_mem_partition_ids_accessor[i];
            int64_t edge_bucket_size = edge_bucket_sizes_accessor[edge_bucket_id];
            int64_t edge_bucket_start = edge_bucket_starts_disk_accessor[edge_bucket_id];

            int idx = offset + i * num_swap_partitions + j;
            new_edge_bucket_ids_accessor[idx] = edge_bucket_id;
            new_edge_bucket_sizes_accessor[idx] = edge_bucket_size;
            new_global_edge_bucket_starts_accessor[idx] = edge_bucket_start;
        }
    }

    // concatenate old and new
    in_mem_edge_bucket_ids = torch::cat({in_mem_edge_bucket_ids, new_edge_bucket_ids});
    in_mem_edge_bucket_sizes = torch::cat({in_mem_edge_bucket_sizes, new_edge_bucket_sizes});
    local_or_global_edge_bucket_starts = torch::cat({local_or_global_edge_bucket_starts, new_global_edge_bucket_starts});

    torch::Tensor in_mem_mask = torch::ones({num_edge_buckets_in_mem - num_new_edge_buckets}, torch::kBool);
    in_mem_mask = torch::cat({in_mem_mask, torch::zeros({num_new_edge_buckets}, torch::kBool)});

    // put the ids in the correct order so the mapped edges remain sorted
    torch::Tensor src_ids_order = torch::zeros({num_edge_buckets_in_mem}, torch::kInt64);
    auto src_ids_order_accessor = src_ids_order.accessor<int64_t, 1>();

#pragma omp parallel for
    for (int i = 0; i < buffer_size; i++) {
        for (int j = 0; j < buffer_size; j++) {
            int64_t edge_bucket_id = new_in_mem_partition_ids_accessor[i] * num_partitions + new_in_mem_partition_ids_accessor[j];

            int idx = i * buffer_size + j;
            src_ids_order_accessor[idx] = edge_bucket_id;
        }
    }

    // TODO: all these argsorts can be done with one omp for loop, probably faster, same with masked_selects above
    torch::Tensor arg_sort = torch::argsort(in_mem_edge_bucket_ids);
    arg_sort = (arg_sort.index_select(0, torch::argsort(torch::argsort(src_ids_order))));
    in_mem_edge_bucket_ids = (in_mem_edge_bucket_ids.index_select(0, arg_sort));
    in_mem_edge_bucket_sizes = (in_mem_edge_bucket_sizes.index_select(0, arg_sort));
    local_or_global_edge_bucket_starts = (local_or_global_edge_bucket_starts.index_select(0, arg_sort));
    in_mem_mask = (in_mem_mask.index_select(0, arg_sort));

    // with everything in order grab the edge buckets
    torch::Tensor in_mem_edge_bucket_starts = in_mem_edge_bucket_sizes.cumsum(0);
    int64_t total_size = in_mem_edge_bucket_starts[-1].item<int64_t>();
    in_mem_edge_bucket_starts = in_mem_edge_bucket_starts - in_mem_edge_bucket_sizes;

    auto in_mem_edge_bucket_sizes_accessor = in_mem_edge_bucket_sizes.accessor<int64_t, 1>();
    auto local_or_global_edge_bucket_starts_accessor = local_or_global_edge_bucket_starts.accessor<int64_t, 1>();
    auto in_mem_mask_accessor = in_mem_mask.accessor<bool, 1>();
    auto in_mem_edge_bucket_starts_accessor = in_mem_edge_bucket_starts.accessor<int64_t, 1>();

    torch::Tensor new_all_in_memory_edges = torch::empty({total_size, storage_ptrs_.edges->dim1_size_}, torch::kInt64);

// get the edges
#pragma omp parallel for
    for (int i = 0; i < num_edge_buckets_in_mem; i++) {
        int64_t edge_bucket_size = in_mem_edge_bucket_sizes_accessor[i];
        int64_t edge_bucket_start = local_or_global_edge_bucket_starts_accessor[i];
        bool in_mem = in_mem_mask_accessor[i];
        int64_t local_offset = in_mem_edge_bucket_starts_accessor[i];

        if (in_mem) {
            new_all_in_memory_edges.narrow(0, local_offset, edge_bucket_size) =
                current_subgraph_state_->all_in_memory_edges_.narrow(0, edge_bucket_start, edge_bucket_size);
        } else {
            new_all_in_memory_edges.narrow(0, local_offset, edge_bucket_size) = storage_ptrs_.edges->range(edge_bucket_start, edge_bucket_size);
        }
    }

    subgraph->all_in_memory_edges_ = new_all_in_memory_edges;

    if (storage_ptrs_.node_embeddings != nullptr) {
        subgraph->global_to_local_index_map_ =
            std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_embeddings)->getGlobalToLocalMap(!prefetch_);
    } else if (storage_ptrs_.node_features != nullptr) {
        subgraph->global_to_local_index_map_ = std::dynamic_pointer_cast<PartitionBufferStorage>(storage_ptrs_.node_features)->getGlobalToLocalMap(!prefetch_);
    }

    torch::Tensor mapped_edges;
    torch::Tensor mapped_edges_dst_sort;
    if (storage_ptrs_.edges->dim1_size_ == 3) {
        mapped_edges = torch::stack({subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, 0)),
                                     subgraph->all_in_memory_edges_.select(1, 1),
                                     subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, -1))})
                           .transpose(0, 1);
    } else if (storage_ptrs_.edges->dim1_size_ == 2) {
        mapped_edges = torch::stack({subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, 0)),
                                     subgraph->global_to_local_index_map_.index_select(0, subgraph->all_in_memory_edges_.select(1, -1))})
                           .transpose(0, 1);
    } else {
        // TODO use a function for logging errors and throwing expections
        SPDLOG_ERROR("Unexpected number of edge columns");
        std::runtime_error("Unexpected number of edge columns");
    }

    //    assert((mapped_edges == -1).nonzero().size(0) == 0);
    //    assert((mapped_edges_dst_sort == -1).nonzero().size(0) == 0);

    subgraph->all_in_memory_mapped_edges_ = mapped_edges;

    mapped_edges = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, true);
    mapped_edges_dst_sort = merge_sorted_edge_buckets(mapped_edges, in_mem_edge_bucket_starts, buffer_size, false);

    mapped_edges = mapped_edges.to(torch::kInt64);
    mapped_edges_dst_sort = mapped_edges_dst_sort.to(torch::kInt64);

    int num_hash_maps = current_subgraph_state_->in_memory_subgraph_->num_hash_maps_;

    if (subgraph->in_memory_subgraph_ != nullptr) {
        subgraph->in_memory_subgraph_ = nullptr;
    }

    subgraph->in_memory_subgraph_ = std::make_shared<MariusGraph>(mapped_edges, mapped_edges_dst_sort, getNumNodesInMemory(), num_hash_maps);

    // update state
    subgraph->in_memory_partition_ids_ = new_in_mem_partition_ids;
    subgraph->in_memory_edge_bucket_ids_ = in_mem_edge_bucket_ids;
    subgraph->in_memory_edge_bucket_sizes_ = in_mem_edge_bucket_sizes;
    subgraph->in_memory_edge_bucket_starts_ = in_mem_edge_bucket_starts;

    if (prefetch_) {
        prefetch_complete_ = true;
        subgraph_lock_->unlock();
        subgraph_cv_->notify_all();
    }
}

EdgeList GraphModelStorage::merge_sorted_edge_buckets(EdgeList edges, torch::Tensor starts, int buffer_size, bool src) {
    int sort_dim = 0;
    if (!src) {
        sort_dim = -1;
    }
    return edges.index_select(0, torch::argsort(edges.select(1, sort_dim)));
}

void GraphModelStorage::sortAllEdges() {
    if (!useInMemorySubGraph()) {
        std::vector<EdgeList> additional_edges = {};

        if (storage_ptrs_.train_edges != nullptr) {
            storage_ptrs_.train_edges->load();
            additional_edges.emplace_back(storage_ptrs_.train_edges->range(0, storage_ptrs_.train_edges->getDim0()));
        }

        if (storage_ptrs_.validation_edges != nullptr) {
            storage_ptrs_.validation_edges->load();
            additional_edges.emplace_back(storage_ptrs_.validation_edges->range(0, storage_ptrs_.validation_edges->getDim0()));
        }

        if (storage_ptrs_.test_edges != nullptr) {
            storage_ptrs_.test_edges->load();
            additional_edges.emplace_back(storage_ptrs_.test_edges->range(0, storage_ptrs_.test_edges->getDim0()));
        }

        for (auto f_edges : storage_ptrs_.filter_edges) {
            f_edges->load();
            additional_edges.emplace_back(f_edges->range(0, f_edges->getDim0()));
        }

        current_subgraph_state_->in_memory_subgraph_->sortAllEdges(torch::cat(additional_edges));

        for (auto f_edges : storage_ptrs_.filter_edges) {
            f_edges->unload();
        }
    } else {
        current_subgraph_state_->in_memory_subgraph_->sortAllEdges(current_subgraph_state_->in_memory_subgraph_->src_sorted_edges_);
    }
}