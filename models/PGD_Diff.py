import pytorch_lightning as pl
from torch.nn.functional import mse_loss
from tqdm import tqdm
from modules.sequence.encode import *
from modules.sequence.transformer import SeqTransformer
from modules.structure.egnn import EGNN, DualPathStructureEncoder
from modules.structure.encode import *
from util.constant import seq_length_freq, get_seq_constant_init
from util.diffusion_util import get_para_schedule, clip_norm
from util.embed.embedding import structure_embedding, sequence_embedding
from util.embed.sequence import index_to_fasta
from util.geometry import Peptide
from typing import Optional

class PGD_Diff(pl.LightningModule):
    def __init__(
            self,
            # parameters for structure diffusion
            struct_node_input_dim=46,
            struct_node_hidden_dim=128,
            struct_edge_dim=8,
            struct_edge_hidden_dim=32,
            struct_node_output_dim=128,
            struct_n_layer=4,
            # parameters for sequence diffusion
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
            # 新增：是否使用双路径结构编码器
            use_dual_path_structure: bool = True,
            # 双路径编码器专属参数
            global_hidden_dim: int = 128,
            global_n_head: int = 4,
            global_n_layers: int = 2,
            fusion: str = 'gate',
            # ========== 新增开关 ==========
            use_contrastive: bool = True,   # 是否启用对比学习
    ):
        super().__init__()

        self.learning_rate_struct = learning_rate_struct
        self.learning_rate_seq = learning_rate_seq
        self.learning_rate_cont = learning_rate_cont
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.use_contrastive = use_contrastive  # 保存开关

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

        # 边特征编码器（共用）
        self.edge_attr_mlp = MLPEdgeEncoder(
            edge_dim=struct_edge_dim,
            output_dim=struct_edge_hidden_dim
        )

        # 结构编码器
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

        # 坐标噪声预测头
        self.struct_ffn = StructFFN(
            input_dim=struct_node_output_dim,
            hidden_dim=struct_node_hidden_dim
        )

        # sequence diffusion
        self.n_class = seq_n_class
        self.clamp = seq_clamp
        self.transformer = SeqTransformer(
            input_dim=proj_dim,
            output_dim=seq_n_hidden,
            n_block=seq_n_blocks,
            use_dual_path=True,
            local_kernel_size=3
        )
        self.seq_ffn = SeqFFN(seq_n_hidden, seq_n_class)

        # ========== 对比学习模块（条件初始化） ==========
        if self.use_contrastive:
            self.struct_attention = SelfAttention(
                n_emb=struct_node_output_dim,
                n_head=n_self_atte_head
            )
            self.seq_attention = SelfAttention(
                n_emb=seq_n_hidden,
                n_head=n_self_atte_head
            )
            self.sentence_predictor = MetricPredictorLayer(input_dim=seq_n_hidden)
            self.seq_predictor = MetricPredictorLayer(input_dim=seq_n_hidden)
            self.graph_predictor = MetricPredictorLayer(input_dim=struct_node_output_dim)
            self.struct_predictor = MetricPredictorLayer(input_dim=struct_node_output_dim)
            self.metric_loss = MetricLoss(temperature=temperature)
            self.match_loss = MatchLoss(temperature=temperature)
        else:
            # 设置为 None，后续调用时检查
            self.struct_attention = None
            self.seq_attention = None
            self.sentence_predictor = None
            self.seq_predictor = None
            self.graph_predictor = None
            self.struct_predictor = None
            self.metric_loss = None
            self.match_loss = None

    def get_loss(self, batch):
        batch_size = len(batch.x)
        time_step = torch.ones(batch_size, device=self.device, dtype=torch.int64) * self.time_sampler.sample()

        # structure diffusion (ACP)
        pos_noise_pred, acp_graph_cont_emb, acp_graph_match_emb, pos_0, pos_t, pos_noise_t, a_pos, acp_struct_emb = \
            self.struct_forward(batch, time_step, "ACP")
        # structure diffusion (nonACP)
        _, nonacp_graph_cont_emb, *_, nonacp_struct_emb = \
            self.struct_forward(batch, time_step, "nonACP")

        # sequence diffusion (ACP)
        acp_x0_real, acp_x0_pred, acp_sentence_cont_emb, acp_sentence_match_emb = self.seq_forward(
            time_step, batch, batch_size, "ACP", struct_emb=acp_struct_emb)
        # sequence diffusion (nonACP)
        _, _, nonacp_sentence_cont_emb, _ = self.seq_forward(
            time_step, batch, batch_size, "nonACP", struct_emb=nonacp_struct_emb)

        # structure loss
        struct_pos_loss = mse_loss(pos_noise_pred, pos_noise_t)
        pred_pos_0 = (1. / a_pos).sqrt() * (pos_t - (1.0 - a_pos).sqrt() * pos_noise_pred)
        struct_pred_loss = torch.sqrt(mse_loss(pred_pos_0, pos_0))
        self.log("struct_score", struct_pred_loss, prog_bar=True)

        # sequence loss
        seq_kl_loss = multinomial_kl(acp_x0_pred, acp_x0_real)
        seq_pred_score = token_aa_acc(acp_x0_pred, acp_x0_real, self.device)
        self.log("seq_score", seq_pred_score, prog_bar=True)

        diff_loss = struct_pos_loss + seq_kl_loss
        self.log("diff_loss", diff_loss, prog_bar=True)

        # ========== 对比损失（仅当启用时计算） ==========
        if self.use_contrastive:
            struct_metric_loss = self.metric_loss(acp_graph_cont_emb, nonacp_graph_cont_emb)
            seq_metric_loss = self.metric_loss(acp_sentence_cont_emb, nonacp_sentence_cont_emb)
            intra_loss = struct_metric_loss + seq_metric_loss
            inter_loss = self.match_loss(acp_graph_match_emb, acp_sentence_match_emb, match_type="graph")
            contrast_loss = intra_loss + inter_loss
            self.log("contrast_loss", contrast_loss, prog_bar=True)
            total_loss = self.loss_weight * diff_loss + (1 - self.loss_weight) * contrast_loss
        else:
            # 无对比学习，仅扩散损失
            contrast_loss = torch.tensor(0.0, device=self.device)
            total_loss = diff_loss   # 或 self.loss_weight * diff_loss 根据需要，这里直接 diff_loss
            # 仍记录一个零值以便日志统一
            self.log("contrast_loss", contrast_loss, prog_bar=True)

        self.log("total_loss", total_loss, prog_bar=True)

        return total_loss, pred_pos_0, pos_0, acp_x0_pred, acp_x0_real

    def struct_pred(self, x, edge_index, edge_attr, edge_length, pos, batch, time_step):
        edge_attr = self.edge_attr_mlp(edge_attr=edge_attr, edge_length=edge_length)
        node_emb = self.structure_encoder(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_length=edge_length,
            batch=batch,
            time_step=time_step,
            pos=pos
        )
        pos_noise_pred = self.struct_ffn(node_emb, time_step, batch)
        return pos_noise_pred, node_emb

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

        # 根据 use_contrastive 和 diff_statue 决定是否计算对比嵌入
        if self.use_contrastive and diff_statue:
            node_emb, attn = self.struct_attention(node_emb, batch=batch_index)
            graph_emb = get_attn_emb(node_emb, attn, batch_index)
            graph_emb = torch.concat(graph_emb, dim=0)
            graph_cont_emb = self.struct_predictor(graph_emb)
            graph_match_emb = self.graph_predictor(graph_emb)
        else:
            graph_cont_emb = None
            graph_match_emb = None

        return pos_noise_pred, graph_cont_emb, graph_match_emb, pos, pos_t, pos_noise_t, a_pos, node_emb

    def seq_pred(self, seq_data, time_step, batch, struct_emb=None):
        seq_emb = self.transformer(seq_data, time_step, batch, struct_emb=struct_emb)
        output = self.seq_ffn(seq_emb)
        seq_pred = F.softmax(output, dim=-1).float()
        return seq_pred, seq_emb

    def seq_forward(self, seq_time_steps, batch, batch_size, seq_type, diff_statue: bool = True, struct_emb=None):
        assert seq_type in {"ACP", "nonACP"}, "seq_type error"
        if seq_type == "ACP":
            x0_real = batch.logit
            batch_index = get_batch_info(batch.fasta, self.device)
        else:
            x0_real = batch.nonacp_logit
            batch_index = get_batch_info(batch.nonacp_fasta, self.device)

        token_time_steps = seq_time_steps.index_select(0, batch_index)
        alphas_bar = self.alphas_bar.index_select(0, seq_time_steps)

        noise = get_seq_noise(device=self.device)
        Qt_weight = get_Qt_weight(alphas_bar, noise, batch_index, self.device, self.n_class)
        x_t = torch.matmul(x0_real.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        x_t_emb = batch_sequence_embedding(x_t, batch_index, batch_size, self.device)
        x_t_emb = self.projection(x_t_emb)

        x0_pred, token_emb = self.seq_pred(x_t_emb, token_time_steps, batch_index, struct_emb=struct_emb)

        if self.use_contrastive and diff_statue:
            token_emb, attn = self.seq_attention(token_emb, batch=batch_index)
            sentence_emb = get_attn_emb(token_emb, attn, batch_index)
            sentence_emb = torch.concat(sentence_emb, dim=0)
            sentence_cont_emb = self.seq_predictor(sentence_emb)
            sentence_match_emb = self.sentence_predictor(sentence_emb)
        else:
            sentence_cont_emb = None
            sentence_match_emb = None

        return x0_real, x0_pred, sentence_cont_emb, sentence_match_emb

    def q_posterior(self, x0, time_step, batch):
        # 与原始代码相同，省略...
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
    def denoise_struct_sample(self, fasta_seq, pdb_out_statue=False, pdb_name=None):
        # 与原始代码相同，省略...
        pos_init = torch.randn((len(fasta_seq), 4, 3), device=self.device)
        pos = pos_init
        pos_traj = []
        t_list = torch.arange(self.num_timestep - 1, 0, -1).to(self.device)
        for i in tqdm(range(len(t_list))):
            node_embedding, edge_index, edge_attr, edge_length = structure_embedding(
                pos, fasta=fasta_seq, constant_data=self.seq_constant_data)
            pos_noise, _ = self.struct_pred(node_embedding, edge_index, edge_attr, edge_length, pos,
                                            batch=None, time_step=t_list[i])
            pos_noise = pos_noise.reshape(-1, 4, 3)
            alpha_bar_t = self.alphas_bar[t_list[i]]
            alpha_t = self.alphas[t_list[i]]
            beta_t = self.betas[t_list[i]]
            mean_eps = (1. / alpha_t).sqrt() * (pos - beta_t * pos_noise / (1. - alpha_bar_t).sqrt())
            log_var = beta_t.log()
            noise = torch.randn_like(mean_eps, device=self.device)
            pos = mean_eps + torch.exp(0.5 * log_var) * noise
            pos_traj.append(pos[:, 0, :].clone().cpu())
        if pdb_out_statue:
            peptide = Peptide(pos, fasta_seq)
            peptide.reconstruct()
            peptide.output_to_pdb(pdb_name or fasta_seq[:5])
        return pos, pos_traj

    @torch.no_grad()
    def denoise_seq_sample(self, n_seq=1, seq_length=None, fasta_out_statue=False):
        # 与原始代码相同，省略...
        seq_freq = torch.tensor(seq_length_freq, device=self.device)
        D = torch.distributions.Categorical(seq_freq)
        out_seq_list = []
        out_seq_traj = []
        for i in range(n_seq):
            seq_len = seq_length[i] if seq_length is not None else D.sample()
            seq_init = get_seq_noise(seq_len, self.device)
            seq_index_t = logit_to_index(seq_init, random_state=True)
            batch = torch.zeros(seq_len, device=self.device).long()
            t_list = torch.arange(self.num_timestep - 1, 0, -1).to(self.device)
            print(f"denoise {i+1}-th sequence")
            for time_steps in tqdm(t_list):
                seq_emb = sequence_embedding(index=seq_index_t)
                seq_emb = torch.tensor(seq_emb, device=self.device).float()
                seq_emb = self.projection(seq_emb)
                token_time_steps = time_steps.repeat(seq_len)
                seq0_pred, _ = self.seq_pred(seq_emb, token_time_steps, batch)
                seq_t = self.q_posterior(seq0_pred, token_time_steps, batch)
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
        total_loss, pred_pos_0, pos_0, acp_x0_pred, acp_x0_real = self.get_loss(batch)
        self.log("train/loss", total_loss)
        return {"loss": total_loss, "pred_pos_0": pred_pos_0, "pos_0": pos_0,
                "acp_x0_pred": acp_x0_pred, "acp_x0_real": acp_x0_real}

    def training_epoch_end(self, training_step_outputs):
        epoch_pred_pos_0 = torch.cat([step["pred_pos_0"] for step in training_step_outputs], dim=0)
        epoch_pos_0 = torch.cat([step["pos_0"] for step in training_step_outputs], dim=0)
        epoch_struct_pred_loss = torch.sqrt(mse_loss(epoch_pred_pos_0, epoch_pos_0))
        self.log("total_struct_loss", epoch_struct_pred_loss, prog_bar=True)

        epoch_acp_x0_pred = torch.cat([step["acp_x0_pred"] for step in training_step_outputs], dim=0)
        epoch_acp_x0_real = torch.cat([step["acp_x0_real"] for step in training_step_outputs], dim=0)
        epoch_seq_pred_score = token_aa_acc(epoch_acp_x0_pred, epoch_acp_x0_real, self.device)
        self.log("total_seq_score", epoch_seq_pred_score, prog_bar=True)

    def configure_optimizers(self):
        # 只添加存在的参数（对比模块可能为 None）
        param_groups = [
            {'params': self.edge_attr_mlp.parameters(), 'lr': self.learning_rate_struct},
            {'params': self.structure_encoder.parameters(), 'lr': self.learning_rate_struct},
            {'params': self.struct_ffn.parameters(), 'lr': self.learning_rate_struct},
            {'params': self.transformer.parameters(), 'lr': self.learning_rate_seq},
            {'params': self.seq_ffn.parameters(), 'lr': self.learning_rate_seq},
        ]
        # 仅当启用对比学习时添加对比模块参数
        if self.use_contrastive:
            param_groups.extend([
                {'params': self.struct_attention.parameters(), 'lr': self.learning_rate_cont},
                {'params': self.seq_attention.parameters(), 'lr': self.learning_rate_cont},
                {'params': self.sentence_predictor.parameters(), 'lr': self.learning_rate_cont},
                {'params': self.graph_predictor.parameters(), 'lr': self.learning_rate_cont},
                {'params': self.seq_predictor.parameters(), 'lr': self.learning_rate_cont},
                {'params': self.struct_predictor.parameters(), 'lr': self.learning_rate_cont},
            ])
        return torch.optim.Adam(param_groups)