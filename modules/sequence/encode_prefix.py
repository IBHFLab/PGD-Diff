
import math
import time
import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord
from torch import nn
import torch.nn.functional as F
import torch
from util.constant import aa_count_freq
from util.embed.embedding import sequence_embedding
from pytorch_metric_learning import miners, distances, losses, reducers
from Bio.Seq import Seq
from Bio import SeqIO

# ==================== 修改后的 SelfAttention（支持 Pair Bias） ====================
class SelfAttention(nn.Module):
    def __init__(self, n_emb, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert n_emb % n_head == 0
        self.n_head = n_head
        self.head_dim = n_emb // n_head
        self.key = nn.Linear(n_emb, n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_emb, n_emb)
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)
        self.proj = nn.Linear(n_emb, n_emb)

    def forward(self, x, batch, pair_bias=None, extra_kv=None):
        token_list, attn_list = [], []
        seq_length_list = torch.bincount(batch)
        n_samples = len(seq_length_list)

        # 🔥 调试：打印 extra_kv 状态
        #if extra_kv is not None:
        #    print(f"DEBUG SelfAttn: extra_kv provided with {len(extra_kv)} items")
        #else:
        #    print("DEBUG SelfAttn: extra_kv is None")

        for i in range(n_samples):
            mask = batch == i
            x_i = x[mask]          # (L_i, n_emb)
            L = x_i.size(0)

            # 原始 Q, K, V
            q = self.query(x_i).view(L, self.n_head, self.head_dim).transpose(0, 1)   # (head, L, d)
            k = self.key(x_i).view(L, self.n_head, self.head_dim).transpose(0, 1)
            v = self.value(x_i).view(L, self.n_head, self.head_dim).transpose(0, 1)

            # 拼接额外的前缀 Key/Value
            if extra_kv is not None and i < len(extra_kv):
                extra_k_i, extra_v_i = extra_kv[i]   
                if extra_k_i is not None and extra_k_i.numel() > 0:
                    # 🔥 调试：打印拼接前的形状
                    #print(f"DEBUG Concat: Sample {i}, Before Cat K: {k.shape}, PK: {extra_k_i.shape}")
                    
                    pk = extra_k_i.view(-1, self.n_head, self.head_dim).transpose(0, 1)  # (head, prefix_len, d)
                    pv = extra_v_i.view(-1, self.n_head, self.head_dim).transpose(0, 1)
                    
                    k = torch.cat([pk, k], dim=1)   # (head, prefix_len + L, d)
                    v = torch.cat([pv, v], dim=1)
                    
                    # 🔥 调试：打印拼接后的形状
                    #print(f"DEBUG Concat: Sample {i}, After Cat K: {k.shape}")
                else:
                    print(f"⚠️ WARNING: Sample {i} has empty prefix!")
            else:
                print(f"⚠️ WARNING: Sample {i} skipped prefix! extra_kv={extra_kv is not None}, len={len(extra_kv) if extra_kv else 0}")
                
            # 计算注意力
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

            # 添加 pair bias（需要扩展到包含 prefix 的维度）
            if pair_bias is not None and i < len(pair_bias):
                bias_i = pair_bias[i]   # (L, L)
                # 仅对原始序列部分添加 bias，前缀部分无 bias
                # 用零填充前缀部分
                total_len = k.size(1)
                pad = total_len - L
                if pad > 0:
                    bias_i = F.pad(bias_i, (0, pad, 0, pad), value=0.0)
                bias_i = bias_i.unsqueeze(0).expand(self.n_head, -1, -1)
                attn = attn + bias_i

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)

            out = attn @ v   # (head, L, d)
            out = out.transpose(0, 1).contiguous().view(L, self.n_head * self.head_dim)
            attn = attn.mean(dim=0)   # (total_len, total_len) 均值池化后的注意力矩阵

            out = self.resid_drop(self.proj(out))
            token_list.append(out)
            attn_list.append(attn)

        y = torch.cat(token_list, dim=0)
        return y, attn_list

# ==================== 以下所有内容保持原样 ====================

class CrossAttention(nn.Module):
    def __init__(self,
                 n_emb,
                 n_cond_emb,
                 n_head,
                 attn_drop=0.1,
                 resid_drop=0.1,
                 ):
        super().__init__()

        self.key = nn.Linear(n_cond_emb, n_emb)
        self.query = nn.Linear(n_emb, n_emb)
        self.value = nn.Linear(n_cond_emb, n_emb)
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)
        self.proj = nn.Linear(n_emb, n_emb)
        self.n_head = n_head

    def forward(self, x, condition, batch=None):
        T, C = x.size()
        T_cond, _ = condition.size()
        k = self.key(condition).view(T_cond, self.n_head, C // self.n_head).transpose(0, 1)
        q = self.query(x).view(T, self.n_head, C // self.n_head).transpose(0, 1)
        v = self.value(condition).view(T_cond, self.n_head, C // self.n_head).transpose(0, 1)

        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)
        y = attention @ v
        y = y.transpose(1, 2).contiguous().view(T, C)
        attention = attention.mean(dim=0, keepdim=False)

        y = self.resid_drop(self.proj(y))
        return y, attention


class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=2000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SeqFFN(nn.Module):
    def __init__(self, input_dim, output_dim, activation="silu", dropout=0.1):
        super().__init__()
        self.dim_list = [input_dim, input_dim * 4, input_dim * 2, input_dim, input_dim // 2, input_dim // 4, output_dim]
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = None
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.layers = nn.ModuleList()
        for i in range(len(self.dim_list) - 1):
            self.layers.append(nn.Linear(self.dim_list[i], self.dim_list[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                if self.activation:
                    x = self.activation(x)
                if self.dropout:
                    x = self.dropout(x)
        return x


class MetricLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.miner = miners.MultiSimilarityMiner()
        self.distances = distances.CosineSimilarity()
        self.reducer = reducers.MeanReducer()
        self.metric_trip_loss = losses.TripletMarginLoss(distance=self.distances, reducer=self.reducer)
        self.metric_cont_loss = losses.ContrastiveLoss(distance=self.distances, pos_margin=1, neg_margin=0)
        self.cont_loss = ContrastiveLoss(temperature=temperature)

    def forward(self, ACP_emb, nonACP_emb, loss_type="cont"):
        assert loss_type in {"cont", "metric_cont", "metric_trip"}, print("metric_loss_type error")
        emb = torch.cat((ACP_emb, nonACP_emb), dim=0)
        ACP_label = torch.ones(len(ACP_emb), device=ACP_emb.device)
        nonACP_label = torch.zeros(len(nonACP_emb), device=nonACP_emb.device)
        label = torch.cat((ACP_label, nonACP_label))

        if loss_type == "cont":
            loss = self.cont_loss(emb, label)
        elif loss_type == "metric_cont":
            loss = self.metric_cont_loss(emb, label)
        else:
            hard_pairs = self.miner(emb, label)
            loss = self.metric_trip_loss(emb, label, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        n = labels.shape[0]
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        mask_pos = torch.ones_like(similarity_matrix, device=features.device) * (
            labels.expand(n, n).eq(labels.expand(n, n).t()))
        mask_neg = torch.ones_like(mask_pos, device=features.device) - mask_pos
        similarity_matrix = torch.exp(similarity_matrix / self.T)
        mask_diag = (torch.ones(n, n) - torch.eye(n, n)).to(features.device)
        similarity_matrix = similarity_matrix * mask_diag
        sim_pos = mask_pos * similarity_matrix
        sim_neg = similarity_matrix - sim_pos
        sim_neg = torch.sum(sim_neg, dim=1).repeat(n, 1).T
        sim_total = sim_pos + sim_neg
        loss = torch.div(sim_pos, sim_total)
        loss = mask_neg + loss + torch.eye(n, n, device=features.device)
        loss = -torch.log(loss)
        loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)
        return loss


class MatchLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, feature_left, feature_right, match_type="graph"):
        assert match_type in {"node", "graph"}, print("match_type error")
        device = feature_left.device
        if match_type == "node":
            similarity = F.cosine_similarity(feature_left, feature_right, dim=1).to(device)
            similarity = torch.exp(similarity / self.T)
            loss = torch.mean(-torch.log(similarity))
        else:
            n = len(feature_left)
            similarity = F.cosine_similarity(feature_left.unsqueeze(1), feature_right.unsqueeze(0), dim=2).to(device)
            similarity = torch.exp(similarity / self.T)
            mask_pos = torch.eye(n, n, device=device, dtype=bool)
            sim_pos = torch.masked_select(similarity, mask_pos)
            sim_total_row = torch.sum(similarity, dim=0)
            loss_row = torch.div(sim_pos, sim_total_row)
            loss_row = -torch.log(loss_row)
            sim_total_col = torch.sum(similarity, dim=1)
            loss_col = torch.div(sim_pos, sim_total_col)
            loss_col = -torch.log(loss_col)
            loss = loss_row + loss_col
            loss = torch.sum(loss) / (2 * n)
        return loss


class MetricPredictorLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.activate = nn.ReLU()
        self.project_emb = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            self.activate,
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            self.activate,
            nn.Linear(hidden_dim * 2, input_dim)
        )

    def forward(self, feature):
        out = self.project_emb(feature)
        return out


def index_to_onehot(x, num_classes=20):
    x = torch.tensor(x)
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    return x_onehot.float()


def logit_to_index(logit_p, random_state=False):
    if random_state:
        D = torch.distributions.Categorical(logit_p)
        token_index = D.sample()
    else:
        token_index = logit_p.argmax(dim=-1)
    return token_index


def multinomial_kl(prob1, prob2):
    prob1 = prob1.softmax(dim=-1)
    prob2 = prob2.softmax(dim=-1)
    kl = (prob1 * torch.log(prob1 / prob2)).sum(dim=-1)
    return kl.mean()


def get_time_steps(n_seq, n_timestep, device=None):
    time_step = torch.randint(1, n_timestep, size=(n_seq,), device=device)
    return time_step


def get_seq_noise(seq_len=1, device=None, noise_state="dmd", n_class=20):
    if noise_state == "dud":
        noise = torch.ones([seq_len, n_class], device=device) / n_class
    else:
        noise = torch.tensor(aa_count_freq, device=device).unsqueeze(dim=0).repeat(seq_len, 1)
    return noise


def get_Qt_weight(alphas_bar, noise, batch, device, n_class=20):
    Qt_weight = [bar_t * torch.eye(n_class, device=device) + (1 - bar_t) * noise for bar_t in alphas_bar]
    Qt_weight = torch.stack(Qt_weight).float()
    Qt_weight = Qt_weight.index_select(0, batch)
    return Qt_weight


def batch_sequence_embedding(seq_logit, batch, batch_size, device):
    seq_emd_list = []
    for i in range(batch_size):
        seq_index = logit_to_index(seq_logit[batch == i,])
        seq_emd = sequence_embedding(index=seq_index)
        seq_emd_list.append(seq_emd)
    seq_emd_list = np.concatenate(seq_emd_list, axis=0)
    out = torch.tensor(seq_emd_list, device=device).float()
    return out


def token_aa_acc(pred, real, device):
    y_pred = torch.argmax(pred, dim=-1)
    y_real = torch.argmax(real, dim=-1)
    score = torch.sum((y_pred == y_real).to(device)) / len(y_pred)
    return score


def calculate_peptide_properties(fasta_list):
    charge_map = {'R': 1, 'K': 1, 'H': 0.1, 'D': -1, 'E': -1}
    hydro_map = {'A': 0.25, 'R': -1.8, 'N': -0.64, 'C': 0.04, 'Q': -0.69, 'G': 0.16,
                 'H': -0.4, 'I': 0.73, 'L': 0.53, 'K': -1.1, 'M': 0.26, 'F': 0.61,
                 'P': -0.07, 'S': -0.26, 'T': -0.18, 'W': 0.37, 'Y': 0.02, 'V': 0.54}
    batch_charges = []
    batch_hydros = []
    for seq in fasta_list:
        charge = sum([charge_map.get(aa, 0) for aa in seq])
        hydro = sum([hydro_map.get(aa, 0) for aa in seq]) / len(seq)
        batch_charges.append(charge)
        batch_hydros.append(hydro)
    return torch.tensor(batch_charges), torch.tensor(batch_hydros)


def get_seq_batch_info(data, device):
    nonACP_fasta_list = data.nonacp_seq
    y_batch = []
    for i in range(len(nonACP_fasta_list)):
        seq_length = len(nonACP_fasta_list[i])
        seq_value = torch.full([seq_length], i, device=device)
        y_batch.append(seq_value)
    y_batch = torch.concat(y_batch)
    data.y_batch = y_batch
    return data


def get_batch_info(data_list, device):
    batch = []
    for i in range(len(data_list)):
        data_length = len(data_list[i])
        data_value = torch.full([data_length], i, device=device)
        batch.append(data_value)
    batch = torch.concat(batch)
    return batch


def get_struct_batch_info(fasta_list, device):
    batch = []
    for i in range(len(fasta_list)):
        fasta_length = len(fasta_list[i])
        fasta_value = torch.full([fasta_length], i, device=device)
        batch.append(fasta_value)
    batch = torch.concat(batch)
    return batch


def get_attn_emb(seq_emb, seq_attn, batch):
    seq_size = len(seq_attn)
    seq_attn_emb_list = []
    for index in range(seq_size):
        attn = seq_attn[index].mean(dim=0)
        emb = seq_emb[batch == index,]
        attn_emb = attn @ emb
        attn_emb = attn_emb.unsqueeze(0)
        seq_attn_emb_list.append(attn_emb)
    return seq_attn_emb_list


def save_output_seq(out_seq_list):
    record_list = []
    for i, seq_str in enumerate(out_seq_list):
        record = SeqRecord(Seq(seq_str), id=f"seq_{i}", description="")
        record_list.append(record)
    record_path = "data/output/fasta/generated_sequences.fasta"
    print("save to " + record_path)
    SeqIO.write(record_list, record_path, "fasta")
    return record_path