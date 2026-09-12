# PGD_Diff_Condition_Final.py
# 修改说明：
# 1. 修复推理采样：引入 Temperature, Top-p 和 Length Temperature 解决模式与长度坍缩
# 2. 修复训练逻辑：引入 Condition Drop (15%) 确保 CFG 有效
# 3. 移除自动冻结逻辑，交由训练脚本统一管理

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import mse_loss
from tqdm import tqdm

# 假设这些模块存在于你的项目中
from modules.sequence.encode import *
from modules.sequence.transformer_prefix import SeqTransformer
from modules.structure.egnn import EGNN, DualPathStructureEncoder
from modules.structure.encode import *
from util.constant import seq_length_freq, get_seq_constant_init
from util.diffusion_util import get_para_schedule, clip_norm
from util.embed.embedding import structure_embedding, sequence_embedding
from util.embed.sequence import index_to_fasta
from util.geometry import Peptide

def sample_condition_subset(cancer_multi_hot, training=True):
    if not training:
        return cancer_multi_hot.clone()
    batch_size = cancer_multi_hot.size(0)
    new_multi_hot = torch.zeros_like(cancer_multi_hot)
    for b in range(batch_size):
        active_indices = torch.where(cancer_multi_hot[b] > 0)[0].tolist()
        if not active_indices:
            continue
        rand_val = torch.rand(1).item()
        if rand_val < 0.4:
            chosen_idx = np.random.choice(active_indices, 1).tolist()
        elif rand_val < 0.7:
            if len(active_indices) >= 2:
                k = np.random.randint(2, len(active_indices) + 1)
                chosen_idx = np.random.choice(active_indices, k, replace=False).tolist()
            else:
                chosen_idx = active_indices
        else:
            chosen_idx = active_indices
        for idx in chosen_idx:
            new_multi_hot[b, idx] = 1.0
    return new_multi_hot


class PGD_Diff(pl.LightningModule):
    def __init__(
            self,
            struct_node_input_dim=46,
            struct_node_hidden_dim=128,
            struct_edge_dim=8,
            struct_edge_hidden_dim=32,
            struct_node_output_dim=128,
            struct_n_layer=4,
            seq_n_class=20,
            seq_n_seq_emb=2072,
            proj_dim=256,
            seq_n_hidden=128,
            seq_clamp=-50,
            seq_n_blocks=8,
            n_timestep=200,
            n_self_atte_head=4,
            beta_schedule="linear",
            beta_start=1.e-7,
            beta_end=2.e-2,
            temperature=0.1,
            learning_rate_struct=5e-3,
            learning_rate_seq=5e-3,
            learning_rate_cont=5e-3,
            loss_weight=0.9,
            use_dual_path_structure=True,
            global_hidden_dim=128,
            global_n_head=4,
            global_n_layers=2,
            fusion='gate',
            num_cancer_types=21,
            condition_mode='hybrid',
            prefix_len=10,
    ):
        super().__init__()

        self.learning_rate_struct = learning_rate_struct
        self.learning_rate_seq = learning_rate_seq
        self.learning_rate_cont = learning_rate_cont
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.time_sampler = torch.distributions.Categorical(torch.ones(n_timestep))
        self.seq_constant_data = get_seq_constant_init(self.device)

        self.proj_dim = proj_dim
        self.projection = nn.Linear(seq_n_seq_emb, proj_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        betas, alphas, alphas_bar = get_para_schedule(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timestep=n_timestep,
            device=self.device
        )
        self.betas = nn.Parameter(betas, requires_grad=False)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.alphas_bar = nn.Parameter(alphas_bar, requires_grad=False)
        self.num_timestep = n_timestep

        self.edge_attr_mlp = MLPEdgeEncoder(
            edge_dim=struct_edge_dim,
            output_dim=struct_edge_hidden_dim
        )

        self.use_dual_path_structure = use_dual_path_structure
        if self.use_dual_path_structure:
            self.structure_encoder = DualPathStructureEncoder(
                node_input_dim=struct_node_input_dim,
                node_hidden_dim=struct_node_hidden_dim,
                edge_dim=struct_edge_hidden_dim,
                node_output_dim=struct_node_output_dim,
                num_layer=struct_n_layer,
                global_hidden_dim=global_hidden_dim,
                global_n_head=global_n_head,
                global_n_layers=global_n_layers,
                fusion=fusion
            )
        else:
            self.structure_encoder = EGNN(
                node_input_dim=struct_node_input_dim,
                node_hidden_dim=struct_node_hidden_dim,
                edge_dim=struct_edge_hidden_dim,
                node_output_dim=struct_node_output_dim,
                num_layer=struct_n_layer
            )

        self.struct_ffn = StructFFN(
            input_dim=struct_node_output_dim,
            hidden_dim=struct_node_hidden_dim
        )

        self.n_class = seq_n_class
        self.clamp = seq_clamp

        # ==================== 条件控制模块 ====================
        self.num_cancer_types = num_cancer_types
        self.condition_mode = condition_mode
        self.prefix_len = prefix_len

        self.cancer_embedding = nn.Embedding(num_cancer_types, proj_dim)
        nn.init.normal_(self.cancer_embedding.weight, std=0.02)
        
        self.condition_mlp = nn.Sequential(
            nn.Linear(proj_dim, proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim)
        )
        
        self.uncond_embedding = nn.Parameter(torch.randn(proj_dim) * 0.02)

        # ==================== Sequence Transformer ====================
        self.transformer = SeqTransformer(
            input_dim=proj_dim,
            output_dim=seq_n_hidden,
            n_block=seq_n_blocks,
            use_dual_path=True,
            local_kernel_size=3,
            num_cancers=num_cancer_types,
            prefix_len=prefix_len
        )
        
        self.seq_ffn = SeqFFN(seq_n_hidden, seq_n_class)

        # Contrastive Learning Modules
        self.struct_attention = SelfAttention(n_emb=struct_node_output_dim, n_head=n_self_atte_head)
        self.seq_attention = SelfAttention(n_emb=seq_n_hidden, n_head=n_self_atte_head)
        self.sentence_predictor = MetricPredictorLayer(input_dim=seq_n_hidden)
        self.seq_predictor = MetricPredictorLayer(input_dim=seq_n_hidden)
        self.graph_predictor = MetricPredictorLayer(input_dim=struct_node_output_dim)
        self.struct_predictor = MetricPredictorLayer(input_dim=struct_node_output_dim)
        self.metric_loss = MetricLoss(temperature=temperature)
        self.match_loss = MatchLoss(temperature=temperature)

        # Classifier for Auxiliary Loss
        self.classifier = nn.Linear(seq_n_hidden, num_cancer_types)
        self._aa_embedding = nn.Parameter(torch.randn(seq_n_class, seq_n_hidden) * 0.02)

    def get_aa_embedding(self):
        return self._aa_embedding

    def _get_global_condition_bias(self, cancer_multi_hot, batch_size):
        weighted_emb = torch.matmul(cancer_multi_hot, self.cancer_embedding.weight) 
        label_counts = cancer_multi_hot.sum(dim=1, keepdim=True).clamp(min=1.0)
        normalized_emb = weighted_emb / label_counts
        cond_bias = self.condition_mlp(normalized_emb) 
        return cond_bias

    def get_loss(self, batch):
        batch_size = len(batch.fasta)
        cancer_multi_hot = batch.cancer_multi_hot.to(self.device)
        
        # 🔥 关键修改 1: 训练时引入 Condition Drop (15% 概率置零)
        # 这是 Classifier-Free Guidance (CFG) 能够生效的核心前提
        if self.training and torch.rand(1).item() < 0.15:
            cancer_multi_hot = torch.zeros_like(cancer_multi_hot)
            
        time_step = torch.ones(batch_size, device=self.device, dtype=torch.int64) * self.time_sampler.sample()
        
        acp_x0_real, acp_x0_pred, _, _, seq_emb = self.seq_forward(
            time_step, batch, batch_size, "ACP", struct_emb=None,
            cancer_multi_hot=cancer_multi_hot, diff_statue=False
        )
        
        seq_kl_loss = multinomial_kl(acp_x0_pred, acp_x0_real)
        seq_pred_score = token_aa_acc(acp_x0_pred, acp_x0_real, self.device)
        
        aa_emb = self.get_aa_embedding()
        soft_seq_emb = torch.matmul(acp_x0_pred, aa_emb)
        
        batch_index = batch.batch_index.to(self.device)
        seq_emb_pooled = []
        for b in range(batch_size):
            mask = (batch_index == b)
            if mask.sum() > 0:
                pooled = soft_seq_emb[mask].mean(dim=0)
                seq_emb_pooled.append(pooled)
            else:
                seq_emb_pooled.append(torch.zeros(soft_seq_emb.size(1), device=soft_seq_emb.device))
        seq_emb_pooled = torch.stack(seq_emb_pooled)
        
        logits = self.classifier(seq_emb_pooled)
        
        valid_mask = (cancer_multi_hot.sum(dim=1) > 0)
        if valid_mask.any():
            cls_loss = F.binary_cross_entropy_with_logits(
                logits[valid_mask], cancer_multi_hot[valid_mask]
            )
        else:
            cls_loss = torch.tensor(0.0, device=logits.device)
            
        CLS_WEIGHT = 1.0
        total_loss = seq_kl_loss + CLS_WEIGHT * cls_loss
        
        self.log("seq_score", seq_pred_score, prog_bar=True, batch_size=batch_size)
        self.log("cls_loss", cls_loss, prog_bar=True, batch_size=batch_size)
        self.log("total_loss", total_loss, prog_bar=True, batch_size=batch_size)
        
        return total_loss, None, None, acp_x0_pred, acp_x0_real

    def seq_pred(self, seq_data, time_step, batch, struct_emb=None, cancer_multi_hot=None, cond_emb=None, return_logits=False):
        batch_size = cancer_multi_hot.shape[0] if cancer_multi_hot is not None else 1
        seq_data_cond = seq_data 
            
        seq_emb = self.transformer(
            seq_data_cond, time_step, batch, struct_emb=struct_emb,
            cancer_multi_hot=cancer_multi_hot,
            cond_emb=cond_emb
        )
        
        logits = self.seq_ffn(seq_emb)
        
        if return_logits:
            return logits, seq_emb
            
        seq_pred = F.softmax(logits, dim=-1).float()
        return seq_pred, seq_emb

    def seq_forward(self, seq_time_steps, batch, batch_size, seq_type, diff_statue=True, struct_emb=None, cancer_multi_hot=None):
        if seq_type == "ACP":
            x0_real = batch.logit.to(self.device)
            batch_index = batch.batch_index.to(self.device)
        else:
            x0_real = batch.nonacp_logit.to(self.device)
            batch_index = get_batch_info(batch.nonacp_fasta, self.device)

        alphas_bar = self.alphas_bar.index_select(0, seq_time_steps)
        noise = get_seq_noise(device=self.device)
        Qt_weight = get_Qt_weight(alphas_bar, noise, batch_index, self.device, self.n_class)
        x_t = torch.matmul(x0_real.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)
        x_t_emb = batch_sequence_embedding(x_t, batch_index, batch_size, self.device)
        x_t_emb = self.projection(x_t_emb)

        cond_emb = None
        if cancer_multi_hot is not None and self.condition_mode in ['hybrid', 'global_only']:
            cond_emb = self._get_global_condition_bias(cancer_multi_hot, batch_size)

        token_time_steps = seq_time_steps.index_select(0, batch_index)
        
        x0_pred, token_emb = self.seq_pred(
            x_t_emb, token_time_steps, batch_index,
            struct_emb=struct_emb, 
            cancer_multi_hot=cancer_multi_hot,
            cond_emb=cond_emb
        )
        
        if diff_statue:
            token_emb, attn = self.seq_attention(token_emb, batch=batch_index)
            sentence_emb = get_attn_emb(token_emb, attn, batch_index)
            sentence_emb = torch.concat(sentence_emb, dim=0)
            sentence_cont_emb = self.seq_predictor(sentence_emb)
            sentence_match_emb = self.sentence_predictor(sentence_emb)
        else:
            sentence_cont_emb = None
            sentence_match_emb = None

        return x0_real, x0_pred, sentence_cont_emb, sentence_match_emb, token_emb

    def struct_forward(self, batch, time_step, struct_type, diff_statue: bool = True):
        assert struct_type in {"ACP", "nonACP"}, "struct_type error"
        alphas_bar = self.alphas_bar.index_select(0, time_step)

        if struct_type == "ACP":
            pos = batch.pos
            fasta_list = batch.fasta
        else:
            pos = batch.nonacp_pos
            fasta_list = batch.nonacp_fasta

        batch_index = get_batch_info(fasta_list, self.device)
        a_pos = alphas_bar.index_select(0, batch_index).unsqueeze(-1).unsqueeze(-1)

        pos_noise_t = torch.randn_like(pos, device=self.device)
        pos_t = a_pos.sqrt() * pos + pos_noise_t * (1.0 - a_pos).sqrt()

        node_emb, edge_index, edge_attr, edge_length = get_batch_structure_embedding(
            pos_t, batch_index, fasta_list, self.device, self.seq_constant_data)

        pos_noise_pred, node_emb = self.struct_pred(node_emb, edge_index, edge_attr, edge_length, pos_t,
                                                    batch_index, time_step)
        pos_noise_pred = clip_norm(pos_noise_pred).reshape(-1, 4, 3)

        if diff_statue:
            node_emb, attn = self.struct_attention(node_emb, batch=batch_index)
            graph_emb = get_attn_emb(node_emb, attn, batch_index)
            graph_emb = torch.concat(graph_emb, dim=0)
            graph_cont_emb = self.struct_predictor(graph_emb)
            graph_match_emb = self.graph_predictor(graph_emb)
        else:
            graph_cont_emb = None
            graph_match_emb = None

        return pos_noise_pred, graph_cont_emb, graph_match_emb, pos, pos_t, pos_noise_t, a_pos, node_emb
    
    def struct_pred(self, node_emb, edge_index, edge_attr, edge_length, pos_t, batch_index, time_step):
         pass

    def q_posterior(self, x0, time_step, batch):
        time_step = (time_step + (self.num_timestep + 1)) % (self.num_timestep + 1)
        alphas = self.alphas.index_select(0, time_step)
        alphas_bar_t = self.alphas_bar.index_select(0, time_step)
        alphas_bar_t_1 = self.alphas_bar.index_select(0, time_step - 1)
        noise = get_seq_noise(device=self.device)

        Qt_weight = get_Qt_weight(alphas_bar_t, noise, batch, self.device, self.n_class)
        xt_from_x0 = torch.matmul(x0.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        Qt_weight = get_Qt_weight(alphas, noise, batch, self.device, self.n_class)
        xt_from_xt_1 = torch.matmul(x0.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        Qt_weight = get_Qt_weight(alphas_bar_t_1, noise, batch, self.device, self.n_class)
        xt_1_from_x0 = torch.matmul(x0.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        xt_1_from_xt = torch.log(x0) - torch.log(xt_from_x0) + torch.log(xt_from_xt_1) + torch.log(xt_1_from_x0)
        xt_1_from_xt = torch.clamp(xt_1_from_xt, self.clamp, 0)
        xt_1_from_xt = torch.exp(xt_1_from_xt)
        return xt_1_from_xt

    @torch.no_grad()
    def denoise_seq_sample_with_prefix(self, cancer_multi_hot=None, n_seq=1, seq_length=None,
                                        fasta_out_statue=False, guidance_scale=1.5, 
                                        temperature=1.2, top_p=0.9, length_temperature=1.0):
        """
        推理采样函数
        :param guidance_scale: CFG 强度 (推荐 1.2 - 2.0)
        :param temperature: 采样温度，>1.0 增加多样性，<1.0 增加确定性 (推荐 1.0 - 1.5)
        :param top_p: Top-p (Nucleus) 采样阈值，截断长尾概率 (推荐 0.85 - 0.95)
        :param length_temperature: 长度分布平滑温度，>1.0 使长度分布更均匀，缓解长度坍缩
        """
        seq_freq = torch.tensor(seq_length_freq, device=self.device)
        
        # 🔥 关键修改 2: 平滑长度分布，解决长度坍缩
        if length_temperature > 1.0:
            adjusted_freq = seq_freq ** (1.0 / length_temperature)
            adjusted_freq = adjusted_freq / adjusted_freq.sum()
            D = torch.distributions.Categorical(adjusted_freq)
        else:
            D = torch.distributions.Categorical(seq_freq)
            
        out_seq_list, out_seq_traj = [], []

        if cancer_multi_hot is not None:
            if isinstance(cancer_multi_hot, list):
                cancer_multi_hot = torch.tensor(cancer_multi_hot, device=self.device, dtype=torch.float32)
            elif isinstance(cancer_multi_hot, torch.Tensor):
                cancer_multi_hot = cancer_multi_hot.to(self.device)
            else:
                raise ValueError("cancer_multi_hot must be a tensor or list of multi-hot vectors")
            if cancer_multi_hot.shape[0] != n_seq:
                if cancer_multi_hot.shape[0] == 1:
                    cancer_multi_hot = cancer_multi_hot.repeat(n_seq, 1)
                else:
                    raise ValueError(f"cancer_multi_hot shape[0] ({cancer_multi_hot.shape[0]}) != n_seq ({n_seq})")
        else:
            cancer_multi_hot = torch.zeros(n_seq, self.num_cancer_types, device=self.device)

        for i in range(n_seq):
            seq_len = int(seq_length[i]) if seq_length is not None else D.sample()
            seq_init = get_seq_noise(seq_len, self.device)
            seq_index_t = logit_to_index(seq_init, random_state=True)
            batch = torch.zeros(seq_len, device=self.device).long()
            t_list = torch.arange(self.num_timestep - 1, 0, -1).to(self.device)
            cur_multi_hot = cancer_multi_hot[i].unsqueeze(0)

            for time_steps in tqdm(t_list, leave=False):
                seq_emb = sequence_embedding(index=seq_index_t)
                seq_emb = torch.tensor(seq_emb, device=self.device).float()
                seq_emb = self.projection(seq_emb)

                if seq_emb.shape[0] != seq_len:
                    if seq_emb.shape[0] > seq_len:
                        seq_emb = seq_emb[:seq_len]
                    else:
                        pad = torch.zeros(seq_len - seq_emb.shape[0], seq_emb.shape[1], device=seq_emb.device)
                        seq_emb = torch.cat([seq_emb, pad], dim=0)

                cur_cond_emb = None
                if self.condition_mode in ['hybrid', 'global_only']:
                    cur_cond_emb = self._get_global_condition_bias(cur_multi_hot, 1)

                logits_cond, _ = self.seq_pred(
                    seq_emb, time_steps.repeat(seq_len), batch,
                    cancer_multi_hot=cur_multi_hot,
                    cond_emb=cur_cond_emb,
                    return_logits=True
                )
                
                uncond_multi_hot = torch.zeros(1, self.num_cancer_types, device=self.device)
                
                uncond_cond_emb = None
                if self.condition_mode in ['hybrid', 'global_only']:
                    uncond_cond_emb = self._get_global_condition_bias(uncond_multi_hot, 1)

                logits_uncond, _ = self.seq_pred(
                    seq_emb, time_steps.repeat(seq_len), batch,
                    cancer_multi_hot=uncond_multi_hot,
                    cond_emb=uncond_cond_emb,
                    return_logits=True
                )
                
                # Classifier-Free Guidance
                guided_logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
                
                # 🔥 关键修改 3: 引入 Temperature 和 Top-p 采样，解决高频基序坍缩
                if temperature != 1.0:
                    guided_logits = guided_logits / temperature
                    
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(guided_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    guided_logits = guided_logits.masked_fill(indices_to_remove, float('-inf'))

                seq0_pred = F.softmax(guided_logits, dim=-1).float()

                if seq0_pred.shape[0] != seq_len:
                    if seq0_pred.shape[0] > seq_len:
                        seq0_pred = seq0_pred[:seq_len]
                    else:
                        pad = torch.zeros(seq_len - seq0_pred.shape[0], seq0_pred.shape[1], device=seq0_pred.device)
                        seq0_pred = torch.cat([seq0_pred, pad], dim=0)

                seq_t = self.q_posterior(seq0_pred, time_steps.repeat(seq_len), batch)
                seq_index_t = logit_to_index(seq_t, random_state=True)
                out_seq_traj.append(index_to_fasta(seq_index_t))

            seq_fasta = index_to_fasta(seq_index_t)
            out_seq_list.append(seq_fasta)

        if fasta_out_statue:
            record_path = save_output_seq(out_seq_list)
        else:
            record_path = None
        return out_seq_list, out_seq_traj, record_path

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            mh = batch['cancer_multi_hot']
        else:
            mh = batch.cancer_multi_hot
            
        if batch_idx % 10 == 0:
            print(f"[DEBUG] Batch {batch_idx}: multi_hot sum={mh.sum().item():.2f}, "
                f"non_zero_samples={(mh.sum(dim=1)>0).sum().item()}/{mh.size(0)}")
        
        total_loss, pred_pos_0, pos_0, acp_x0_pred, acp_x0_real = self.get_loss(batch)
        self.log("train/loss", total_loss)
        return {"loss": total_loss, "pred_pos_0": pred_pos_0, "pos_0": pos_0,
                "acp_x0_pred": acp_x0_pred, "acp_x0_real": acp_x0_real}

    def training_epoch_end(self, training_step_outputs):
        if training_step_outputs[0]['pred_pos_0'] is not None:
            epoch_pred_pos_0 = torch.cat([s["pred_pos_0"] for s in training_step_outputs], dim=0)
            epoch_pos_0 = torch.cat([s["pos_0"] for s in training_step_outputs], dim=0)
            self.log("total_struct_loss", torch.sqrt(mse_loss(epoch_pred_pos_0, epoch_pos_0)), prog_bar=True)

            epoch_acp_x0_pred = torch.cat([s["acp_x0_pred"] for s in training_step_outputs], dim=0)
            epoch_acp_x0_real = torch.cat([s["acp_x0_real"] for s in training_step_outputs], dim=0)
            self.log("total_seq_score", token_aa_acc(epoch_acp_x0_pred, epoch_acp_x0_real, self.device), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate_seq)
