import numpy as np
from pathlib import Path
import os

import torch  # isort:skip
from marius.tools.preprocess.converters.partitioners.partitioner import Partitioner


def dataframe_to_tensor(df):
    return torch.tensor(df.to_numpy())


def partition_edges(edges, num_nodes, num_partitions, edge_weights=None):
    partition_size = int(np.ceil(num_nodes / num_partitions))

    src_partitions = torch.div(edges[:, 0], partition_size, rounding_mode="trunc")
    dst_partitions = torch.div(edges[:, -1], partition_size, rounding_mode="trunc")

    _, dst_args = torch.sort(dst_partitions, stable=True)
    _, src_args = torch.sort(src_partitions[dst_args], stable=True)
    sort_order = dst_args[src_args]

    edges = edges[sort_order]
    if edge_weights is not None:
        edge_weights = edge_weights[sort_order]

    edge_bucket_ids = torch.div(edges, partition_size, rounding_mode="trunc")
    offsets = np.zeros([num_partitions, num_partitions], dtype=int)
    unique_src, num_source = torch.unique_consecutive(edge_bucket_ids[:, 0], return_counts=True)

    num_source_offsets = torch.cumsum(num_source, 0) - num_source

    curr_src_unique = 0
    for i in range(num_partitions):
        if curr_src_unique < unique_src.size(0) and unique_src[curr_src_unique] == i:
            offset = num_source_offsets[curr_src_unique]
            num_edges = num_source[curr_src_unique]
            dst_ids = edge_bucket_ids[offset : offset + num_edges, -1]

            unique_dst, num_dst = torch.unique_consecutive(dst_ids, return_counts=True)
            offsets[i][unique_dst] = num_dst
            curr_src_unique += 1

    offsets = list(offsets.flatten())
    return edges, offsets, edge_weights


# === ✅ 新增函数：保存 Critical Nodes ===
def select_and_save_critical_nodes(train_edges_tens, num_nodes):
    CRITICAL_NODE_PERCENTILE = 0.02  # 取频率最高的前 2%
    CRITICAL_NODE_UPPER_BOUND = num_nodes  # 可设置为实际最大节点编号
    CRITICAL_NODE_SAVE_PATH = "datasets/ogbn_arxiv/nodes/critical_nodes.bin"

    # 统计出现频率（无向图：源和目的都计入）
    all_nodes = torch.cat([train_edges_tens[:, 0], train_edges_tens[:, 1]])
    node_freq = torch.bincount(all_nodes, minlength=num_nodes)

    # Top-k 高频节点
    num_critical = int(num_nodes * CRITICAL_NODE_PERCENTILE)
    _, topk_indices = torch.topk(node_freq, num_critical, largest=True, sorted=False)
    critical_nodes = topk_indices[topk_indices < CRITICAL_NODE_UPPER_BOUND]

    # 保存为 .bin 文件
    Path(os.path.dirname(CRITICAL_NODE_SAVE_PATH)).mkdir(parents=True, exist_ok=True)
    critical_nodes.to(torch.int64).numpy().tofile(CRITICAL_NODE_SAVE_PATH)
    print(f">>> Saved {len(critical_nodes)} critical nodes to {CRITICAL_NODE_SAVE_PATH}")


# === 原始类结构，加入保存逻辑 ===
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

        train_edges_tens, train_offsets, train_edge_weights = partition_edges(
            train_edges_tens, num_nodes, num_partitions, edge_weights=train_edge_weights
        )

        # ✅ 保存 critical 节点
        select_and_save_critical_nodes(train_edges_tens, num_nodes)

        valid_offsets = None
        test_offsets = None

        if self.partitioned_evaluation:
            if valid_edges_tens is not None:
                valid_edges_tens, valid_offsets, valid_edge_weights = partition_edges(
                    valid_edges_tens, num_nodes, num_partitions, edge_weights=valid_edge_weights
                )

            if test_edges_tens is not None:
                test_edges_tens, test_offsets, test_edge_weights = partition_edges(
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
