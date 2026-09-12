import numpy as np
import os
from Bio import SeqIO
import torch
from torch_geometric.nn import radius_graph
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from util.embed.sequence import *
from util.embed.structure import *
# 注意：sequence 和 structure 模块应提供以下函数：
#   index_to_fasta, onehot_encoding, get_bio_embedding_for_sequence,
#   get_position_from_pdb, get_aaindex_embedding, get_contact_embedding


def sequence_embedding_from_fastaFile(pdb_type):
    """
    从 FASTA 文件生成序列嵌入，并保存为 pickle 文件。
    """
    assert pdb_type in {"ACP", "nonACP"}, "error type for input"

    if pdb_type == "ACP":
        fasta_path = "data/source/fasta/ACP.fasta"
        data_pickle = "data/source/ACP_sequence.pickle"
    else:
        fasta_path = "data/source/fasta/nonACP.fasta"
        data_pickle = "data/source/nonACP_sequence.pickle"

    if os.path.exists(data_pickle):
        with open(data_pickle, 'rb') as f:
            embedding_list = pickle.load(f)
        print(data_pickle + " exists")
        return embedding_list

    fasta_list = SeqIO.parse(fasta_path, "fasta")
    embedding_list = {}

    for fasta in tqdm(fasta_list):
        fasta_id = fasta.id
        fasta_seq = fasta.seq
        # 调用 sequence_embedding 时只传入序列字符串
        sequence_data = sequence_embedding(fasta=fasta_seq)
        embedding_list[fasta_id] = sequence_data

    with open(data_pickle, 'wb') as f:
        print(data_pickle + " save")
        pickle.dump(embedding_list, f)

    return embedding_list


def sequence_embedding(fasta=None, index=None):
    if fasta is None:
        assert index is not None, "sequence_embedding: either fasta or index must be provided"
        fasta = index_to_fasta(index)

    # one-hot 编码 [L, 20]
    embedding_1 = onehot_encoding(fasta)
    # 生物学特征
    embedding_2 = get_bio_embedding_for_sequence(fasta)
    L = embedding_1.shape[0]
    
    # 确保维度一致，模型期望总维度 2072
    expected_total = 2072
    if embedding_1.shape[1] + embedding_2.shape[1] != expected_total:
        print(f"序列 {fasta[:30]}... 总维度 {embedding_1.shape[1] + embedding_2.shape[1]} != {expected_total}")
        # 调整 embedding_2 到 (L, expected_total-20)
        if embedding_2.shape[1] < expected_total - 20:
            padding = np.zeros((L, expected_total - 20 - embedding_2.shape[1]))
            embedding_2 = np.concatenate([embedding_2, padding], axis=1)
        elif embedding_2.shape[1] > expected_total - 20:
            embedding_2 = embedding_2[:, :expected_total-20]
    
    embedding = np.concatenate((embedding_1, embedding_2), axis=1)
    return embedding


def structure_embeddings_from_pdb(pdb_type):
    """ 
    从 PDB 文件提取所有序列的 3D 坐标，并保存为 pickle 文件。
    """
    assert pdb_type in {"ACP", "nonACP"}, "error type for input"

    if pdb_type == "ACP":
        pdb_dir = "data/source/pdb/ACP"
        fasta_path = "data/source/fasta/ACP.fasta"
        data_pickle = "data/source/ACP_structure.pickle"
    else:
        pdb_dir = "data/source/pdb/nonACP"
        fasta_path = "data/source/fasta/nonACP.fasta"
        data_pickle = "data/source/nonACP_structure.pickle"

    if os.path.exists(data_pickle):
        with open(data_pickle, 'rb') as f:
            embedding_list = pickle.load(f)
        print(data_pickle + " exists")
        return embedding_list

    fasta_list = SeqIO.parse(fasta_path, "fasta")
    embedding_list = {}
    for fasta in tqdm(fasta_list):
        fasta_id = fasta.id
        pdb_file = f"{pdb_dir}/{fasta_id}.pdb"
        pos_list = get_position_from_pdb(pdb_file)
        embedding_list[fasta_id] = pos_list

    with open(data_pickle, 'wb') as f:
        pickle.dump(embedding_list, f)

    return embedding_list


def structure_embedding(pos, fasta=None, index=None, threshold=5, device=None, batch=None, constant_data=None):
    """
    根据 3D 坐标和序列构建图结构。
    pos: 原子坐标，形状 (N_residues, 4, 3) 或 (N_residues, 3)
    fasta 或 index: 提供序列信息
    threshold: 距离阈值，用于建边
    device: 张量设备
    batch: 批次索引，用于 radius_graph
    constant_data: 预计算的常量数据（用于接触特征）
    """
    if fasta is None:
        assert index is not None, "structure_embedding: either fasta or index must be provided"
        fasta = index_to_fasta(index)

    # 获取每个残基的 AAIndex 特征（形状 [N, node_dim]）
    node_emb = get_aaindex_embedding(fasta, device)

    # 使用 CA 原子坐标（通常是每个残基的第一个原子）
    if pos.ndim == 3 and pos.shape[1] == 4:
        pos_ca = pos[:, 0, :]
    else:
        pos_ca = pos  # 假设已经是 CA 坐标

    pos_ca = torch.as_tensor(pos_ca, device=device)

    # 构建半径图
    edge_index = radius_graph(pos_ca, r=threshold, batch=batch).long()
    
    # 处理无边的情况
    if edge_index.numel() == 0:
        # 无边：返回空的边索引、边特征和边距离
        edge_length = torch.empty((0, 1), device=device)
        # 根据模型定义，边特征维度为 struct_edge_dim=8
        feat_dim = 8
        edge_emb = torch.empty((0, feat_dim), device=device)
        node_emb = torch.as_tensor(node_emb, device=device).float()
        return node_emb, edge_index, edge_emb, edge_length

    # 有边时计算边距离和接触特征
    edge_length = F.pairwise_distance(pos_ca[edge_index[0]], pos_ca[edge_index[1]], p=2).unsqueeze(dim=-1)

    node_emb = torch.as_tensor(node_emb, device=device).float()
    edge_emb = get_contact_embedding(fasta, edge_index.T, constant_data, edge_length)
    edge_emb = torch.as_tensor(edge_emb, device=device)
    
    # 确保 edge_emb 形状正确：如果是一维，则变为 (num_edges, -1)
    if edge_emb.dim() == 1:
        edge_emb = edge_emb.reshape(-1, 1)
    else:
        edge_emb = edge_emb.reshape(edge_emb.size(0), -1)

    return node_emb, edge_index, edge_emb, edge_length