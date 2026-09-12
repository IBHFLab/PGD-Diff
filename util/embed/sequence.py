import numpy as np
import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from util.constant import *
import json
from typing import Optional, List, Union, Dict

# 全局 ESMC 嵌入缓存
ESMC_EMBEDDINGS: Optional[Dict] = None

def fasta_to_index(fasta: str) -> List[int]:
    return [AA_dict[aa] for aa in fasta]

def index_to_fasta(index_list: List[int]) -> str:
    return "".join(AA_type[index] for index in index_list)

def onehot_encoding(seq):
    if len(seq) == 0:
        return np.zeros((0, len(AA_type)))
    encoding_map = np.eye(len(AA_type))
    residues_map = {}
    for i, aa in enumerate(AA_type):
        residues_map[aa] = encoding_map[i]
    tmp_seq = [residues_map[aa] for aa in seq]
    return np.array(tmp_seq)

def position_encoding(seq_length):
    d = 20
    b = 50
    N = seq_length
    value = []
    for pos in range(N):
        tmp = []
        for i in range(d // 2):
            tmp.append(pos / (b ** (2 * i / d)))
        value.append(tmp)
    value = np.array(value)
    pos_encoding = np.zeros((N, d))
    pos_encoding[:, 0::2] = np.sin(value[:, :])
    pos_encoding[:, 1::2] = np.cos(value[:, :])
    return pos_encoding

def save_index_to_fasta(index_list: List[List[int]], output_path: str = "data/output") -> None:
    os.makedirs(output_path, exist_ok=True)
    records = []
    for i, indices in enumerate(index_list):
        seq = index_to_fasta(indices)
        record = SeqRecord(Seq(seq), id=f"output_fasta_{i}", description="")
        records.append(record)
    SeqIO.write(records, f"{output_path}/output.fasta", "fasta")

def load_fasta_to_index(fasta_path: str) -> List[List[int]]:
    index_list = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        index_list.append(fasta_to_index(str(record.seq)))
    return index_list

# ==================== ESMC 嵌入 ====================
def load_esmc_embeddings(json_path: str = "data/esmc_embeddings_complete.json") -> Dict:
    global ESMC_EMBEDDINGS
    if ESMC_EMBEDDINGS is not None:
        return ESMC_EMBEDDINGS
    try:
        with open(json_path, 'r') as f:
            ESMC_EMBEDDINGS = json.load(f)
        print(f"成功加载 {len(ESMC_EMBEDDINGS)} 个序列的 ESMC 嵌入")
    except Exception as e:
        ESMC_EMBEDDINGS = {}
    return ESMC_EMBEDDINGS

def get_esmc_embedding_from_peptide(fasta: str) -> np.ndarray:
    if len(fasta) == 0:
        return np.zeros((0, 1152))
    global ESMC_EMBEDDINGS
    if ESMC_EMBEDDINGS is None:
        ESMC_EMBEDDINGS = load_esmc_embeddings()
    
    if fasta in ESMC_EMBEDDINGS:
        emb = ESMC_EMBEDDINGS[fasta]
        emb = np.array(emb)
        if emb.ndim == 1:
            # 检查一维向量长度是否为1152，否则填充/截断
            if emb.shape[0] != 1152:
                print(f"警告：ESMC嵌入维度为 {emb.shape[0]}，期望1152，将截断/填充")
                if emb.shape[0] > 1152:
                    emb = emb[:1152]
                else:
                    emb = np.pad(emb, (0, 1152 - emb.shape[0]), 'constant')
            emb = np.tile(emb, (len(fasta), 1))
        else:
            # 二维情况，检查第二维是否为1152
            if emb.shape[1] != 1152:
                print(f"警告：ESMC嵌入维度为 {emb.shape[1]}，期望1152，将截断/填充")
                if emb.shape[1] > 1152:
                    emb = emb[:, :1152]
                else:
                    padding = np.zeros((emb.shape[0], 1152 - emb.shape[1]))
                    emb = np.concatenate([emb, padding], axis=1)
        return emb
    else:
        return np.zeros((len(fasta), 1152))

# ==================== BLOSUM 嵌入 ====================
def get_blosum_embedding_from_peptide(fasta: str) -> np.ndarray:
    if len(fasta) == 0:
        return np.zeros((0, 20))
    embedding = np.array([blosum62[aa] for aa in fasta])
    return embedding

# ==================== PAM 嵌入 ====================
def get_pam_embedding_from_peptide(fasta: str) -> np.ndarray:
    if len(fasta) == 0:
        return np.zeros((0, 20))
    embedding = np.array([PAM120[aa] for aa in fasta])
    return embedding

# ==================== 疏水性嵌入 ====================
def get_hydrophobicity_embedding_from_peptide(fasta, scale=100):
    if len(fasta) == 0:
        return np.zeros((0, 20))
    flat_list = []
    for aa in fasta:
        flat_list.extend([score / scale for score in hydrophobicity[aa]])
    embedding = np.array(flat_list)
    embedding = embedding.reshape(len(fasta), -1)
    return embedding


# ==================== AAC 嵌入 ====================
def get_aac_embedding_from_peptide(fasta: str) -> np.ndarray:
    if len(fasta) == 0:
        return np.zeros((0, 20))
    L = len(fasta)
    counts = np.zeros(len(AA_type))
    for aa in fasta:
        idx = AA_dict[aa]
        counts[idx] += 1
    freq = counts / L
    embedding = np.tile(freq, (L, 1))
    return embedding

# ==================== AARPC 嵌入 ====================
def get_aarpc_embedding_from_peptide(fasta: str, n_segments: int = 5) -> np.ndarray:
    if len(fasta) == 0:
        return np.zeros((0, 20))
    L = len(fasta)
    seg_size = L / n_segments
    seg_counts = np.zeros((n_segments, len(AA_type)))
    for i, aa in enumerate(fasta):
        seg_idx = min(int(i / seg_size), n_segments - 1)
        aa_idx = AA_dict[aa]
        seg_counts[seg_idx, aa_idx] += 1
    seg_freq = np.zeros_like(seg_counts, dtype=float)
    for seg in range(n_segments):
        total = seg_counts[seg].sum()
        if total > 0:
            seg_freq[seg] = seg_counts[seg] / total
    embedding = np.zeros((L, len(AA_type)))
    for i in range(L):
        seg_idx = min(int(i / seg_size), n_segments - 1)
        embedding[i] = seg_freq[seg_idx]
    return embedding

# ==================== CKSAPP 嵌入 ====================
def get_cksapp_embedding_from_peptide(fasta: str, k_max: int = 2) -> np.ndarray:
    if len(fasta) == 0:
        return np.zeros((0, k_max * 400))
    L = len(fasta)
    n_aa = len(AA_type)
    pair_features = []
    for k in range(1, k_max + 1):
        pair_counts = np.zeros((n_aa, n_aa))
        total_pairs = 0
        for i in range(L - k):
            aa1 = fasta[i]
            aa2 = fasta[i + k]
            idx1 = AA_dict[aa1]
            idx2 = AA_dict[aa2]
            pair_counts[idx1, idx2] += 1
            total_pairs += 1
        if total_pairs > 0:
            pair_freq = pair_counts / total_pairs
        else:
            pair_freq = np.zeros((n_aa, n_aa))
        pair_features.append(pair_freq.flatten())
    global_feat = np.concatenate(pair_features)
    embedding = np.tile(global_feat, (L, 1))
    return embedding

# ==================== 通用特征分发 ====================
def get_embedding_from_peptide(fasta: str, encode_type: str) -> np.ndarray:
    if encode_type == "pssm":
        return get_pssm_embedding_from_peptide(fasta)
    elif encode_type == "blosum":
        return get_blosum_embedding_from_peptide(fasta)
    elif encode_type == "pam":
        return get_pam_embedding_from_peptide(fasta)
    elif encode_type == "hydrophobicity":
        return get_hydrophobicity_embedding_from_peptide(fasta)
    elif encode_type == "esmc":
        return get_esmc_embedding_from_peptide(fasta)
    elif encode_type == "aac":
        return get_aac_embedding_from_peptide(fasta)
    elif encode_type == "aarpc":
        return get_aarpc_embedding_from_peptide(fasta)
    elif encode_type == "cksapp":
        return get_cksapp_embedding_from_peptide(fasta)
    else:
        raise ValueError(f"不支持的编码类型: {encode_type}")

get_embedding_form_peptide = get_embedding_from_peptide

def get_bio_embedding_for_sequence(fasta, encode_type=None):
    if encode_type is None:
        encode_type = ["blosum", "pam", "hydrophobicity", "esmc", "aac", "aarpc", "cksapp"]
    
    if len(fasta) == 0:
        # 需要知道期望维度，取一个假序列计算
        dummy_emb = get_embedding_form_peptide("A", encode_type[0])
        total_dim = sum([get_embedding_form_peptide("A", et).shape[1] for et in encode_type])
        return np.zeros((0, total_dim))
    
    embedding_list = []
    L = len(fasta)
    feature_dims = []
    
    for aa_type in encode_type:
        emb = get_embedding_form_peptide(fasta=fasta, encode_type=aa_type)
        if isinstance(emb, list):
            emb = np.array(emb)
        # 确保二维
        if emb.ndim == 1:
            emb = emb.reshape(L, -1)
        elif emb.ndim == 2 and emb.shape[0] != L:
            if emb.shape[1] == L:
                emb = emb.T
            else:
                raise ValueError(f"特征 {aa_type} 的形状 {emb.shape} 无法对齐到序列长度 {L}")
        feature_dims.append(emb.shape[1])
        embedding_list.append(emb)
    
    lengths = [e.shape[0] for e in embedding_list]
    if len(set(lengths)) != 1:
        raise ValueError(f"特征长度不一致: {lengths}")
    
    embedding = np.concatenate(embedding_list, axis=1)
    
    # 期望的生物学特征维度（从模型参数 seq_n_seq_emb - 20 得来，此处设为 2052）
    expected_bio_dim = 2052
    if embedding.shape[1] != expected_bio_dim:
        print(f"警告：序列 {fasta[:30]}... 的生物学特征维度为 {embedding.shape[1]}，期望 {expected_bio_dim}")
        print(f"各特征维度: {dict(zip(encode_type, feature_dims))}")
        if embedding.shape[1] < expected_bio_dim:
            padding = np.zeros((L, expected_bio_dim - embedding.shape[1]))
            embedding = np.concatenate([embedding, padding], axis=1)
        else:
            embedding = embedding[:, :expected_bio_dim]
    
    return embedding