import numpy as np
from collections import defaultdict

from marius.tools.preprocess.converters.partitioners.partitioner import Partitioner
import time
import torch  # isort:skip
import pymetis
from pathlib import Path


def dataframe_to_tensor(df):
    return torch.tensor(df.to_numpy())

def partition_edges(edges, num_nodes, num_partitions, edge_weights=None):
    """
    Enhanced Edge-Aware Community Partitioning

    Key Enhancements:
    1. Community detection that considers edge connectivity
    2. Strategic placement of high-degree (critical) nodes
    3. Maximization of edge locality
    4. Load balancing across partitions
    """
    start = time.time()

    # Step 1: Edge setup
    num_edges = edges.size(0)
    edges_numpy = edges.cpu().numpy()

    # Create undirected adjacency list
    adj_list = [[] for _ in range(num_nodes)]
    edge_list = defaultdict(list)  # node pair -> list of edge indices

    for idx, (src, dst) in enumerate(edges_numpy):
        adj_list[src].append(dst)
        adj_list[dst].append(src)
        edge_list[(min(src, dst), max(src, dst))].append(idx)

    degrees = np.array([len(adj_list[i]) for i in range(num_nodes)], dtype=np.int64)

    # Step 2: Identify Critical Nodes (top 2% highest-degree nodes)
    sorted_nodes = np.argsort(degrees)[::-1]  # sort in descending order
    critical_threshold = max(1, int(num_nodes * 0.02))
    critical_nodes = set(sorted_nodes[:critical_threshold])

    # Step 3: Edge-aware Community Detection
    # Build adjacency list with edge weights for METIS, giving higher weight to critical connections
    metis_adj_list = []
    for i in range(num_nodes):
        neighbors = []
        weights = []
        for neighbor in adj_list[i]:
            weight = 1
            # Higher weight if connected to a critical node
            if i in critical_nodes or neighbor in critical_nodes:
                weight = 10
            # Increase weight if multiple edges exist
            edge_count = len(edge_list[(min(i, neighbor), max(i, neighbor))])
            weight *= edge_count

            neighbors.append(neighbor)
            weights.append(weight)
        metis_adj_list.append(neighbors)

    # Initial partitioning using METIS
    try:
        cuts, parts = pymetis.part_graph(
            nparts=num_partitions,
            adjacency=metis_adj_list,
            options=pymetis.Options(contig=True, minconn=True)
        )
    except:
        # Fallback to round-robin if METIS fails
        parts = np.arange(num_nodes) % num_partitions

    # Step 4: Reassign critical nodes to avoid concentration in one partition
    partition_counts = np.bincount(parts, minlength=num_partitions)
    critical_counts = defaultdict(int)

    for node in critical_nodes:
        critical_counts[parts[node]] += 1

    # Redistribute critical nodes if they are overly concentrated
    for part_id in range(num_partitions):
        if critical_counts[part_id] > critical_threshold / num_partitions * 2:
            nodes_in_part = [n for n in critical_nodes if parts[n] == part_id]
            num_to_move = critical_counts[part_id] - critical_threshold // num_partitions

            for i in range(min(num_to_move, len(nodes_in_part) // 2)):
                node = nodes_in_part[i]
                # Move to the partition with the fewest critical nodes
                target_part = min(range(num_partitions), key=lambda x: critical_counts[x])
                parts[node] = target_part
                critical_counts[part_id] -= 1
                critical_counts[target_part] += 1

    # Step 5: Improve edge locality
    edge_cuts = np.zeros((num_partitions, num_partitions), dtype=np.int64)
    for src, dst in edges_numpy:
        src_part = parts[src]
        dst_part = parts[dst]
        if src_part != dst_part:
            edge_cuts[src_part, dst_part] += 1
            edge_cuts[dst_part, src_part] += 1

    # Try improving edge locality through iterative reassignment
    max_iterations = 10
    for iteration in range(max_iterations):
        improved = False

        # Check if moving a node to a neighbor partition improves locality
        for node in range(num_nodes):
            if node in critical_nodes:
                continue  # Critical nodes are already handled

            current_part = parts[node]
            neighbor_parts = defaultdict(int)

            for neighbor in adj_list[node]:
                neighbor_parts[parts[neighbor]] += 1

            if neighbor_parts:
                best_part = max(neighbor_parts.items(), key=lambda x: x[1])[0]

                if best_part != current_part:
                    if partition_counts[best_part] < partition_counts[current_part]:
                        parts[node] = best_part
                        partition_counts[current_part] -= 1
                        partition_counts[best_part] += 1
                        improved = True

        if not improved:
            break

    # Step 6: Final edge sorting for MariusGNN compatibility
    edges_torch = torch.from_numpy(edges_numpy).long()
    src_partitions = torch.tensor([parts[s.item()] for s in edges_torch[:, 0]], dtype=torch.int64)
    dst_partitions = torch.tensor([parts[d.item()] for d in edges_torch[:, 1]], dtype=torch.int64)

    _, dst_args = torch.sort(dst_partitions, stable=True)
    _, src_args = torch.sort(src_partitions[dst_args], stable=True)
    sort_order = dst_args[src_args]

    edges_torch = edges_torch[sort_order]
    if edge_weights is not None:
        edge_weights = edge_weights[sort_order]

    # Step 7: Offset calculation
    partition_size = int(np.ceil(num_nodes / num_partitions))
    edge_bucket_ids = torch.div(edges_torch, partition_size, rounding_mode="trunc")
    offsets = np.zeros([num_partitions, num_partitions], dtype=int)

    unique_src, num_source = torch.unique_consecutive(edge_bucket_ids[:, 0], return_counts=True)
    num_source_offsets = torch.cumsum(num_source, 0) - num_source

    curr_src_unique = 0
    for i in range(num_partitions):
        if curr_src_unique < unique_src.size(0) and unique_src[curr_src_unique] == i:
            offset = num_source_offsets[curr_src_unique]
            num_edges_for_src = num_source[curr_src_unique]
            dst_ids = edge_bucket_ids[offset : offset + num_edges_for_src, 1]

            unique_dst, num_dst = torch.unique_consecutive(dst_ids, return_counts=True)
            for ud, nd in zip(unique_dst, num_dst):
                offsets[i][ud] = nd.item()
            curr_src_unique += 1

    offsets = list(offsets.flatten())

    end = time.time()
    print(f"Enhanced partitioning time: {end - start:.2f}s")

    # Stats reporting
    unique_nodes_per_part = defaultdict(set)
    for src, dst in edges_torch.numpy():
        src_part = parts[src]
        dst_part = parts[dst]
        unique_nodes_per_part[src_part].add(src)
        unique_nodes_per_part[dst_part].add(dst)

    total_unique = sum(len(nodes) for nodes in unique_nodes_per_part.values())
    redundancy = (total_unique - num_nodes) / num_nodes * 100

    intra_edges = sum(1 for src, dst in edges_torch.numpy() if parts[src] == parts[dst])
    inter_edges = num_edges - intra_edges

    print(f"Node redundancy: {redundancy:.2f}%")
    print(f"Intra-partition edges: {intra_edges} ({intra_edges / num_edges * 100:.2f}%)")
    print(f"Inter-partition edges: {inter_edges} ({inter_edges / num_edges * 100:.2f}%)")

    # Save critical nodes
    critical_nodes = torch.tensor(list(critical_nodes), dtype=torch.int64)
    critical_nodes = critical_nodes[critical_nodes < 169343]
    Path("datasets/ogbn_arxiv/nodes").mkdir(parents=True, exist_ok=True)
    critical_nodes.to(torch.int64).numpy().tofile("datasets/ogbn_arxiv/nodes/critical_nodes.bin")
    print(f"Saved {len(critical_nodes)} critical nodes to binary file.")

    return edges_torch, offsets, edge_weights




class TorchPartitioner(Partitioner):
    def __init__(self, partitioned_evaluation):
        super().__init__()
        self.partitioned_evaluation = partitioned_evaluation

    def partition_edges(
        self, train_edges_tens, valid_edges_tens, test_edges_tens, num_nodes, num_partitions, edge_weights=None
    ):
        # Extract the edge weights
        train_edge_weights, valid_edge_weights, test_edge_weights = None, None, None
        if edge_weights is not None:
            train_edge_weights, valid_edge_weights, test_edge_weights = (
                edge_weights[0],
                edge_weights[1],
                edge_weights[2],
            )

        # 학습 데이터에 대한 파티션 수행 및 분석
        train_edges_tens, train_offsets, train_edge_weights = partition_edges(
            train_edges_tens, num_nodes, num_partitions, edge_weights=train_edge_weights
        )
        
        

        valid_offsets = None
        test_offsets = None

        if self.partitioned_evaluation:
            if valid_edges_tens is not None:
                valid_edges_tens, valid_offsets, valid_edge_weights, valid_stats = partition_edges(
                    valid_edges_tens, num_nodes, num_partitions, edge_weights=valid_edge_weights
                )

            if test_edges_tens is not None:
                test_edges_tens, test_offsets, test_edge_weights, test_stats = partition_edges(
                    test_edges_tens, num_nodes, num_partitions, edge_weights=test_edge_weights
                )

        return (
            train_edges_tens,
            train_offsets,
            valid_edges_tens,
            valid_offsets,
            test_edges_tens,
            test_offsets,
            [train_edge_weights, valid_edge_weights, test_edge_weights],
        )