import math
from torch import nn
import torch.nn.functional as F
import torch
from modules.sequence.encode_prefix import SelfAttention, SinusoidalPosEmb


class LocalAttention(nn.Module):
    def __init__(self, n_emb, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(n_emb, n_emb, kernel_size=kernel_size,
                               padding=kernel_size//2, groups=n_emb)
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, batch=None):
        if batch is None:
            x = x.unsqueeze(0).transpose(1, 2)
            out = self.conv(x)
            out = out.transpose(1, 2).squeeze(0)
        else:
            unique_batches = batch.unique()
            outputs = []
            for b in unique_batches:
                mask = (batch == b)
                seq_x = x[mask]
                seq_len = seq_x.size(0)
                if seq_len < self.conv.kernel_size[0]:
                    outputs.append(seq_x)
                else:
                    seq_x = seq_x.unsqueeze(0).transpose(1, 2)
                    out = self.conv(seq_x)
                    out = out.transpose(1, 2).squeeze(0)
                    outputs.append(out)
            out = torch.cat(outputs, dim=0)
        out = self.activation(out)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, n_emb, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        self.n_head = n_head
        self.n_emb = n_emb
        self.q = nn.Linear(n_emb, n_emb)
        self.kv = nn.Linear(n_emb, 2 * n_emb)
        self.proj = nn.Linear(n_emb, n_emb)
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

    def forward(self, x_seq, x_struct, batch=None):
        T = x_seq.shape[0]
        q = self.q(x_seq).view(T, self.n_head, self.n_emb // self.n_head).transpose(0, 1)
        kv = self.kv(x_struct).view(T, 2, self.n_head, self.n_emb // self.n_head).permute(2, 1, 0, 3)
        k, v = kv[:, 0], kv[:, 1]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.n_emb // self.n_head) ** 0.5
        if batch is not None:
            mask = batch.unsqueeze(0) != batch.unsqueeze(1)
            attn = attn.masked_fill(mask.unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(0, 1).contiguous().view(T, self.n_emb)
        out = self.proj(out)
        out = self.resid_drop(out)
        return out, attn


class SeqBlock(nn.Module):
    def __init__(self,
                 n_emb,
                 n_head,
                 attn_drop,
                 resid_drop,
                 n_diff_step,
                 n_seq_max,
                 emb_type,
                 struct_dim=None,
                 use_dual_path=True,
                 local_kernel_size=3,
                 num_cancers=0,
                 prefix_len=0,
                 cond_dim=None          # 新增：条件嵌入维度
                 ):
        super().__init__()
        self.prefix_len = prefix_len
        self.use_dual_path = use_dual_path

        self.attn_global = SelfAttention(
            n_emb=n_emb,
            n_head=n_head,
            attn_drop=attn_drop,
            resid_drop=attn_drop
        )

        if use_dual_path:
            self.attn_local = LocalAttention(
                n_emb=n_emb,
                kernel_size=local_kernel_size,
                dropout=attn_drop
            )

        self.cross_attn = CrossAttention(
            n_emb=n_emb,
            n_head=n_head,
            attn_drop=attn_drop,
            resid_drop=attn_drop
        )

        self.struct_proj = nn.Linear(struct_dim, n_emb) if (struct_dim and struct_dim != n_emb) else nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, 2 * n_emb),
            nn.GELU(),
            nn.Linear(2 * n_emb, n_emb),
            nn.Dropout(resid_drop),
        )

        self.ln1 = nn.LayerNorm(n_emb, elementwise_affine=False)
        self.ln_cross = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)

        self.dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.cross_dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

        # 时间步和位置嵌入
        if emb_type == "pos_emb":
            self.emb_t = SinusoidalPosEmb(n_diff_step, n_emb)
            self.emb_pos = SinusoidalPosEmb(n_seq_max, n_emb)
        else:
            self.emb_t = nn.Embedding(n_diff_step, n_emb)
            self.emb_pos = nn.Embedding(n_seq_max, n_emb)

        self.silu = nn.SiLU()
        self.linear_t = nn.Linear(n_emb, n_emb)
        self.linear_pos = nn.Linear(n_emb, n_emb)

        # 新增：条件嵌入投影，用于将 cond_emb 映射到 n_emb
        self.cond_proj = nn.Linear(cond_dim, n_emb) if cond_dim else None

        # 癌症特异前缀：每个癌症有独立的 prefix_k, prefix_v
        if prefix_len > 0 and num_cancers > 0:
            self.prefix_k = nn.Parameter(torch.randn(num_cancers, prefix_len, n_emb) * 0.02)
            self.prefix_v = nn.Parameter(torch.randn(num_cancers, prefix_len, n_emb) * 0.02)
        else:
            self.prefix_k = None
            self.prefix_v = None


    def forward(self, x, time_step, batch, struct_emb=None, pair_bias=None, cancer_idx=None, cond_emb=None):
        # cond_emb: [batch_size, cond_dim] 或 None
        # 时间步和位置编码
        time_emb = self.silu(self.linear_t(self.emb_t(time_step)))
        seq_length_list = torch.bincount(batch)
        pos_emb = [torch.arange(0, sl, device=x.device, dtype=torch.float32) for sl in seq_length_list]
        pos_emb = torch.cat(pos_emb, dim=0)
        pos_emb = self.silu(self.linear_pos(self.emb_pos(pos_emb)))
        x = x + time_emb + pos_emb

        # 如果 cond_emb 存在，将其投影后加到 x 上（每个 token 对应其样本的 cond）
        if self.cond_proj is not None and cond_emb is not None:
            cond = self.cond_proj(cond_emb)  # [batch_size, n_emb]
            cond_per_token = cond[batch]     # [total_len, n_emb]
            x = x + cond_per_token

        # 准备 prefix（若存在）
        extra_kv = None
        if self.prefix_k is not None and cancer_idx is not None:
            batch_size = cancer_idx.size(0)
            extra_kv = []
            for i in range(batch_size):
                c_idx = cancer_idx[i]
                pk = self.prefix_k[c_idx]   # (prefix_len, n_emb)
                pv = self.prefix_v[c_idx]
                extra_kv.append((pk, pv))

        # 全局自注意力（传入 prefix）
        a_global, att_global = self.attn_global(x, batch, pair_bias=pair_bias, extra_kv=extra_kv)
        if self.use_dual_path:
            a_local = self.attn_local(x, batch)
            x = self.dropout(x + a_global + a_local)
        else:
            x = self.dropout(x + a_global)
        x = self.ln1(x)

        # 结构交叉注意力（如果有）
        if struct_emb is not None:
            struct_proj = self.struct_proj(struct_emb)
            cross_out, cross_attn = self.cross_attn(x, struct_proj, batch=batch)
            x = self.cross_dropout(x + cross_out)
            x = self.ln_cross(x)

        # MLP 残差
        x = self.dropout(x + self.mlp(x))
        x = self.ln2(x)

        return x, att_global


class SeqTransformer(nn.Module):
    def __init__(
            self,
            input_dim=None,
            output_dim=128,
            n_emb=128,
            n_head=16,
            attn_drop=0.1,
            resid_drop=0.1,
            n_diff_step=500,
            n_block=8,
            emb_type="pos_emb",
            n_seq_max=50,
            struct_dim=128,
            use_dual_path=True,
            local_kernel_size=3,
            num_cancers=0,
            prefix_len=0,
            cond_dim=None          # 新增：条件嵌入维度（来自癌症嵌入）
    ):
        super().__init__()
        self.cont_emb = nn.Linear(input_dim, n_emb)
        self.n_block = n_block
        self.use_dual_path = use_dual_path

        self.output_emb = nn.Sequential(
            nn.LayerNorm(n_emb),
            nn.Linear(n_emb, output_dim),
        )

        self.blocks = nn.Sequential(*[
            SeqBlock(
                n_emb=n_emb,
                n_head=n_head,
                attn_drop=attn_drop,
                resid_drop=resid_drop,
                n_diff_step=n_diff_step,
                emb_type=emb_type,
                n_seq_max=n_seq_max,
                struct_dim=struct_dim,
                use_dual_path=use_dual_path,
                local_kernel_size=local_kernel_size,
                num_cancers=num_cancers,
                prefix_len=prefix_len,
                cond_dim=cond_dim
            ) for _ in range(n_block)
        ])

    def forward(self, x, time_step, batch=None, struct_emb=None, cond_emb=None, pair_bias=None, cancer_idx=None):
        x_emb = self.cont_emb(x)
        # 注意：cond_emb 已经在外部通过 cancer_idx 获取并投影好，这里直接传入即可
        for block in self.blocks:
            x_emb, _ = block(x_emb, time_step, batch, struct_emb, pair_bias, cancer_idx=cancer_idx, cond_emb=cond_emb)
        return self.output_emb(x_emb)