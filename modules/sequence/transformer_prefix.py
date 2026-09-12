import math
from torch import nn
import torch.nn.functional as F
import torch
from modules.sequence.encode_prefix import SinusoidalPosEmb


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
                 has_prefix=False
                 ):
        super().__init__()
        self.prefix_len = prefix_len
        self.use_dual_path = use_dual_path
        self.num_cancers = num_cancers
        self.n_head = n_head
        self.n_emb = n_emb
        self.has_prefix = has_prefix
        
        self.q_proj = nn.Linear(n_emb, n_emb)
        self.k_proj = nn.Linear(n_emb, n_emb)
        self.v_proj = nn.Linear(n_emb, n_emb)
        self.o_proj = nn.Linear(n_emb, n_emb)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.resid_drop = nn.Dropout(resid_drop)

        if use_dual_path:
            self.attn_local = LocalAttention(n_emb=n_emb, kernel_size=local_kernel_size, dropout=attn_drop)

        self.cross_attn = CrossAttention(n_emb=n_emb, n_head=n_head, attn_drop=attn_drop, resid_drop=resid_drop)
        self.struct_proj = nn.Linear(struct_dim, n_emb) if (struct_dim and struct_dim != n_emb) else nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb), nn.GELU(), nn.Linear(4 * n_emb, 2 * n_emb),
            nn.GELU(), nn.Linear(2 * n_emb, n_emb), nn.Dropout(resid_drop),
        )

        self.ln1 = nn.LayerNorm(n_emb, elementwise_affine=False)
        self.ln_cross = nn.LayerNorm(n_emb)
        self.ln2 = nn.LayerNorm(n_emb)
        self.dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()
        self.cross_dropout = nn.Dropout(attn_drop) if attn_drop > 0 else nn.Identity()

        if emb_type == "pos_emb":
            self.emb_t = SinusoidalPosEmb(n_diff_step, n_emb)
            self.emb_pos = SinusoidalPosEmb(n_seq_max, n_emb)
        else:
            self.emb_t = nn.Embedding(n_diff_step, n_emb)
            self.emb_pos = nn.Embedding(n_seq_max, n_emb)

        self.silu = nn.SiLU()
        self.linear_t = nn.Linear(n_emb, n_emb)
        self.linear_pos = nn.Linear(n_emb, n_emb)

        # ====== Prefix 参数 ======
        if has_prefix and prefix_len > 0 and num_cancers > 0:
            self.prefix_k = nn.Parameter(torch.randn(num_cancers, prefix_len, n_emb) * 0.2)
            self.prefix_v = nn.Parameter(torch.randn(num_cancers, prefix_len, n_emb) * 0.2)
            
            self.uncond_prefix_k = nn.Parameter(torch.randn(prefix_len, n_emb) * 0.2)
            self.uncond_prefix_v = nn.Parameter(torch.randn(prefix_len, n_emb) * 0.2)
            
            self.uncond_ln_k = nn.LayerNorm(n_emb)
            self.uncond_ln_v = nn.LayerNorm(n_emb)
            self.cond_ln_k = nn.LayerNorm(n_emb)
            self.cond_ln_v = nn.LayerNorm(n_emb)
            
            self.prefix_scale = nn.Parameter(torch.ones(1) * 1)
            
        else:
            self.prefix_k = None
            self.prefix_v = None
            self.uncond_prefix_k = None
            self.uncond_prefix_v = None
            self.uncond_ln_k = None
            self.uncond_ln_v = None
            self.cond_ln_k = None
            self.cond_ln_v = None
            self.prefix_scale = None

    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        return torch.matmul(attn_weights, v)

    def forward(self, x, time_step, batch, struct_emb=None, pair_bias=None, cancer_multi_hot=None):
        time_emb = self.silu(self.linear_t(self.emb_t(time_step)))
        seq_length_list = torch.bincount(batch)
        pos_emb = [torch.arange(0, sl, device=x.device, dtype=torch.float32) for sl in seq_length_list]
        pos_emb = torch.cat(pos_emb, dim=0)
        pos_emb = self.silu(self.linear_pos(self.emb_pos(pos_emb)))
        x = x + time_emb + pos_emb

        T = x.shape[0]
        q = self.q_proj(x).view(T, self.n_head, self.n_emb // self.n_head).transpose(0, 1)
        k_seq = self.k_proj(x).view(T, self.n_head, self.n_emb // self.n_head).transpose(0, 1)
        v_seq = self.v_proj(x).view(T, self.n_head, self.n_emb // self.n_head).transpose(0, 1)

        unique_batches = batch.unique()
        outputs = []
        
        for b_idx in unique_batches:
            mask = (batch == b_idx)
            x_b = x[mask]
            q_b = q[:, mask, :]
            k_b = k_seq[:, mask, :]
            v_b = v_seq[:, mask, :]
            
            if self.has_prefix and self.prefix_k is not None and cancer_multi_hot is not None:
                multi_hot = cancer_multi_hot[b_idx]
                n_active = multi_hot.sum()
                
                if n_active > 0:
                    # 使用 LayerNorm 稳定 Cond Prefix
                    cond_pk_raw = torch.einsum('c,cld->ld', multi_hot, self.prefix_k) / n_active
                    cond_pv_raw = torch.einsum('c,cld->ld', multi_hot, self.prefix_v) / n_active
                    
                    pk = self.cond_ln_k(cond_pk_raw)
                    pv = self.cond_ln_v(cond_pv_raw)
                else:
                    # 如果 n_active 为 0，说明最外层传入了无条件分支，直接使用无条件 Prefix
                    pk = self.uncond_ln_k(self.uncond_prefix_k)
                    pv = self.uncond_ln_v(self.uncond_prefix_v)

                pk = pk * self.prefix_scale
                pv = pv * self.prefix_scale
                
                pk = pk.view(self.prefix_len, self.n_head, self.n_emb // self.n_head).transpose(0, 1)
                pv = pv.view(self.prefix_len, self.n_head, self.n_emb // self.n_head).transpose(0, 1)
                
                k_final = torch.cat([pk, k_b], dim=1)
                v_final = torch.cat([pv, v_b], dim=1)
            else:
                k_final = k_b
                v_final = v_b

            out_b = self._scaled_dot_product_attention(q_b, k_final, v_final)
            out_b = out_b.transpose(0, 1).contiguous().view(-1, self.n_emb)
            outputs.append(out_b)
            
        a_global = torch.cat(outputs, dim=0)
        a_global = self.o_proj(a_global)
        a_global = self.resid_drop(a_global)

        if self.use_dual_path:
            a_local = self.attn_local(x, batch)
            x = self.dropout(x + a_global + a_local)
        else:
            x = self.dropout(x + a_global)
        x = self.ln1(x)

        if struct_emb is not None:
            struct_proj = self.struct_proj(struct_emb)
            cross_out, cross_attn = self.cross_attn(x, struct_proj, batch=batch)
            x = self.cross_dropout(x + cross_out)
            x = self.ln_cross(x)

        x = self.dropout(x + self.mlp(x))
        x = self.ln2(x)
        return x, None

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
            prefix_layer_indices=None
    ):
        super().__init__()
        self.cont_emb = nn.Linear(input_dim, n_emb)
        self.n_block = n_block
        self.use_dual_path = use_dual_path
        
        # 解冻 cond_proj 并初始化为非零值，作为条件信号的辅助通道
        self.cond_proj = nn.Linear(input_dim, n_emb)
        nn.init.normal_(self.cond_proj.weight, std=0.01)
        nn.init.zeros_(self.cond_proj.bias)

        self.output_emb = nn.Sequential(nn.LayerNorm(n_emb), nn.Linear(n_emb, output_dim))

        if prefix_layer_indices is None:
            prefix_layer_indices = list(range(n_block))
        
        resolved_indices = set()
        for idx in prefix_layer_indices:
            resolved_indices.add(n_block + idx if idx < 0 else idx)
        
        print(f"🔧 Prefix Tuning Active Layers: {sorted(resolved_indices)} / {n_block}")

        blocks = []
        for i in range(n_block):
            has_prefix = (i in resolved_indices)
            current_prefix_len = prefix_len if has_prefix else 0
            blocks.append(
                SeqBlock(
                    n_emb=n_emb, n_head=n_head, attn_drop=attn_drop, resid_drop=resid_drop,
                    n_diff_step=n_diff_step, emb_type=emb_type, n_seq_max=n_seq_max,
                    struct_dim=struct_dim, use_dual_path=use_dual_path,
                    local_kernel_size=local_kernel_size, num_cancers=num_cancers,
                    prefix_len=current_prefix_len, has_prefix=has_prefix
                )
            )
        self.blocks = nn.Sequential(*blocks)
        
        self._global_step_counter = 0
        self._print_interval = 50

    def _print_all_prefix_grads(self):
        print("\n🔍 [Gradient Monitor] Checking all Prefix Layers:")
        has_any_grad = False
        total_main_grad_norm = 0
        
        # 计算主参数的平均梯度范数作为参考基准
        for name, param in self.named_parameters():
            if 'blocks' in name and 'prefix' not in name and param.grad is not None:
                total_main_grad_norm += param.grad.norm().item()
        
        avg_main_grad = total_main_grad_norm / max(1, len([p for p in self.parameters() if 'blocks' in name and 'prefix' not in name and p.grad is not None]))

        for i, block in enumerate(self.blocks):
            if hasattr(block, 'uncond_prefix_k') and block.uncond_prefix_k is not None:
                k_grad = block.uncond_prefix_k.grad
                v_grad = block.uncond_prefix_v.grad
                
                k_norm = k_grad.norm().item() if k_grad is not None else 0.0
                v_norm = v_grad.norm().item() if v_grad is not None else 0.0
                
                threshold = max(1e-9, avg_main_grad * 0.01)
                
                status_k = "ACTIVE" if k_norm > threshold else "DEAD"
                status_v = "ACTIVE" if v_norm > threshold else "DEAD"
                
                print(f"   Block {i}: K_Norm={k_norm:.6e} ({status_k}), V_Norm={v_norm:.6e} ({status_v})")
                
                if k_norm > threshold or v_norm > threshold:
                    has_any_grad = True
                    
        if not has_any_grad:
            print("   ⚠️ WARNING: All prefix gradients are zero or significantly smaller than main params!")
        print("")

    def forward(self, x, time_step, batch=None, struct_emb=None, cond_emb=None, pair_bias=None, cancer_multi_hot=None):
        x_emb = self.cont_emb(x)
        
        if cond_emb is not None and batch is not None:
            cond_emb_proj = self.cond_proj(cond_emb)
            x_emb = x_emb + cond_emb_proj[batch]
        
        for block in self.blocks:
            x_emb, _ = block(x_emb, time_step, batch, struct_emb, pair_bias, cancer_multi_hot=cancer_multi_hot)
        
        return self.output_emb(x_emb)
    
    def check_prefix_grads(self):
        self._print_all_prefix_grads()