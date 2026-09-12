import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from Bio import SeqIO
import torch
import torch.utils.data
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from modules.data_module.dataSets import *
from util.embed.embedding import structure_embeddings_from_pdb
from util.embed.sequence import onehot_encoding
from util.embed.structure import get_aaindex_embedding
from types import SimpleNamespace

def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_fasta_list(dataType):
    assert dataType in {"ACP", "nonACP"}, print("input type should be ACP or nonACP")
    if dataType == "ACP":
        fasta_path = "data/source/fasta/ACP.fasta"
    else:
        fasta_path = "data/source/fasta/nonACP.fasta"
    fasta_data = SeqIO.parse(fasta_path, "fasta")
    fasta_id_list = []
    fasta_seq_list = []
    for fasta in tqdm(fasta_data):
        fasta_id_list.append(fasta.id)
        fasta_seq_list.append(str(fasta.seq))
    return fasta_id_list, fasta_seq_list

class pgddiffDataModule(pl.LightningDataModule):
    """原始数据模块，用于 ACP/nonACP 数据（使用图数据加载器）"""
    def __init__(self, batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size
        self.ACP_id_list, self.ACP_fasta_list = get_fasta_list("ACP")
        self.nonACP_id_list, self.nonACP_fasta_list = get_fasta_list("nonACP")
        self.ACP_strut_data = structure_embeddings_from_pdb("ACP")
        self.nonACP_strut_data = structure_embeddings_from_pdb("nonACP")
        self.dataset = pgddiffDataset(self.ACP_id_list,
                                   self.ACP_fasta_list,
                                   self.ACP_strut_data,
                                   self.nonACP_id_list,
                                   self.nonACP_fasta_list,
                                   self.nonACP_strut_data)
    def train_dataloader(self):
        return GeometricDataLoader(self.dataset, batch_size=self.batch_size)


# ========== 新增：用于癌症条件微调的数据模块 ==========
def load_cancer_data(csv_path):
    """读取 CSV，返回 DataFrame、癌症类型到索引的映射和类别总数"""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Sequence', 'Cancer Type'])
    all_cancer_types = set()
    for types in df['Cancer Type']:
        for t in str(types).split(','):
            all_cancer_types.add(t.strip())
    cancer_to_idx = {c: i for i, c in enumerate(sorted(all_cancer_types))}
    num_cancer_types = len(cancer_to_idx)
    print(f"共发现 {num_cancer_types} 种癌症类型")
    return df, cancer_to_idx, num_cancer_types

class CancerDataset(torch.utils.data.Dataset):
    """自定义数据集，从 DataFrame 加载序列和癌症标签，并生成 dummy 结构数据"""
    def __init__(self, df, cancer_to_idx, device='cpu'):
        self.sequences = df['Sequence'].tolist()
        self.cancer_labels = df['Cancer Type'].tolist()
        self.cancer_to_idx = cancer_to_idx
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # 序列的 one-hot 编码作为真实分布 logit
        onehot = onehot_encoding(seq)                     # (L, 20)
        logit = torch.tensor(onehot, dtype=torch.float32)

        # 癌症类型 multi-hot 编码
        multi_hot = torch.zeros(len(self.cancer_to_idx), dtype=torch.float32)
        cancer_str = self.cancer_labels[idx]
        if pd.notna(cancer_str):
            for t in str(cancer_str).split(','):
                t = t.strip()
                if t in self.cancer_to_idx:
                    multi_hot[self.cancer_to_idx[t]] = 1.0

        # 生成 dummy 结构数据（用于占位，满足模型输入要求）
        node_emb = get_aaindex_embedding(seq, device=self.device)   # (L, node_dim)
        node_emb = torch.tensor(node_emb, dtype=torch.float32)
        pos = torch.zeros(len(seq), 4, 3, dtype=torch.float32)      # 零坐标

        return logit, seq, multi_hot, node_emb, pos

class ConditionalpgddiffDataModule(pl.LightningDataModule):
    """
    用于癌症肽条件微调的数据模块。
    从 CSV 文件加载序列和癌症标签，并为每个样本生成 dummy 结构数据。
    """
    def __init__(self, csv_path, batch_size=32, num_workers=4, device='cpu'):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

    def setup(self, stage=None):
        df, self.cancer_to_idx, self.num_cancer_types = load_cancer_data(self.csv_path)
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
        self.train_dataset = CancerDataset(train_df, self.cancer_to_idx, self.device)
        self.val_dataset = CancerDataset(val_df, self.cancer_to_idx, self.device)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,                      # 修改为 0
            collate_fn=self.collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,                      # 修改为 0
            collate_fn=self.collate_fn,
            pin_memory=True if torch.cuda.is_available() else False
        )

    def collate_fn(self, batch):
        """
        将 batch 中的样本合并为模型所需的格式。
        返回一个 SimpleNamespace 对象，包含以下属性：
            x: 节点特征 (total_len, node_dim)
            pos: 原子坐标 (total_len, 4, 3)
            logit: 序列真实分布 (total_len, 20)
            fasta: 原始序列列表 (batch_size,)
            nonacp_logit: 占位 (复制 logit)
            nonacp_fasta: 占位 (复制 fasta)
            nonacp_pos: 占位 (复制 pos)
            cancer_label: 癌症标签 (batch_size, num_cancer_types)
        """
        logits, fastas, multi_hots, node_embs, poss = [], [], [], [], []
        batch_idx = []
        for i, (logit, seq, multi_hot, node_emb, pos) in enumerate(batch):
            L = logit.shape[0]
            logits.append(logit)
            fastas.append(seq)
            multi_hots.append(multi_hot)
            node_embs.append(node_emb)
            poss.append(pos)
            batch_idx.extend([i] * L)

        logit_concat = torch.cat(logits, dim=0)                     # (total_len, 20)
        node_emb_concat = torch.cat(node_embs, dim=0)               # (total_len, node_dim)
        pos_concat = torch.cat(poss, dim=0)                         # (total_len, 4, 3)
        batch_idx_tensor = torch.tensor(batch_idx, dtype=torch.long)
        multi_hots_tensor = torch.stack(multi_hots)                 # (batch_size, num_cancer_types)

        batch_obj = SimpleNamespace(
            x=node_emb_concat,
            pos=pos_concat,
            logit=logit_concat,
            fasta=fastas,
            nonacp_logit=logit_concat.clone(),   # 占位
            nonacp_fasta=fastas.copy(),           # 占位
            nonacp_pos=pos_concat.clone(),        # 占位
            cancer_label=multi_hots_tensor        # 新增条件标签
        )
        return batch_obj