# modules/structure/egnn.py
from torch.nn import SiLU, Softplus, Linear, ModuleList, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.norm import LayerNorm
import torch
from torch import nn

from modules.structure.encode import TimeEncoder, atom_pair_feature
#from modules.structure.global_encoder import GlobalStructureEncoder   # 新增导入


class GlobalStructureEncoder(nn.Module):
    """
    轻量级 Transformer 编码器，用于捕获残基间长程依赖。
    输入：残基特征 (N_res, dim) 或坐标 (N_res, 3)
    输出：全局增强的特征 (N_res, dim)
    """
    def __init__(self, input_dim, hidden_dim=128, n_head=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, mask=None):
        """
        x: (N_res, input_dim) 或 (batch, N_res, input_dim)
        mask: (N_res, N_res) 或 None
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (1, N, dim)
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.output_proj(x).squeeze(0)  # (N_res, input_dim)
        x = self.norm(x)
        return x

# equivariant graph convolutional layers (EGCL)等变图卷积层
class EGCL(MessagePassing):
    def __init__(self,
                 node_hidden_dim,
                 edge_dim,
                 atom_edge=165,
                 dropout=0.1,
                 aggr="add",
                 **kwargs
    ):
        assert aggr in {'add', 'sum', 'max', 'mean'}, 'pool method error'
        kwargs.setdefault('aggr', aggr)
        super(EGCL, self).__init__()

        self.atom_edge = atom_edge
        self.m_dim = edge_dim
        self.node_hidden_dim = node_hidden_dim

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        edge_input_dim = 2 * node_hidden_dim + edge_dim + 1

        self.message_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(edge_input_dim * 2, self.m_dim),
            SiLU()
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(self.atom_edge, self.atom_edge * 2),
            SiLU(),
            nn.Linear(self.atom_edge * 2, self.atom_edge),
            SiLU(),
            nn.Linear(self.atom_edge, 3)
        )

        self.edge_weight = nn.Sequential(nn.Linear(self.m_dim, 1), nn.Sigmoid())
        node_input_dim = self.node_hidden_dim + self.m_dim

        self.node_mlp = nn.Sequential(
            nn.Linear(node_input_dim, node_input_dim * 2),
            self.dropout,
            SiLU(),
            nn.Linear(node_input_dim * 2, self.node_hidden_dim),
        )

        self.node_norm = LayerNorm(self.node_hidden_dim)
        self.coors_norm = CoorsNorm()

    def forward(self, x, edge_index, edge_attr, edge_length, batch, pos):
        edge_attr_feats = torch.cat([edge_attr, edge_length], dim=-1)
        node_out, coors_out = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr_feats,
                                             edge_length=edge_length,
                                             batch=batch, pos=pos, size=None)
        return node_out, coors_out

    def message(self, x_i, x_j, edge_attr):
        m_ij = self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return m_ij

    def propagate(self, edge_index, size, **kwargs):
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        m_ij = self.message(**msg_kwargs)

        atom_edge_attr, atom_edge_length = atom_pair_feature(kwargs["x"], edge_index, m_ij, kwargs["pos"])

        coor_wij = self.coors_mlp(atom_edge_attr)
        atom_edge_length = self.coors_norm(atom_edge_length).unsqueeze(dim=-1)
        coors_value = coor_wij * atom_edge_length
        coors_value = coors_value.reshape(-1, 4 * 3)

        mhat_i = self.aggregate(coors_value, **aggr_kwargs)
        mhat_i = mhat_i.reshape(-1, 4, 3)
        coors_out = kwargs["pos"] + mhat_i

        m_ij = m_ij * self.edge_weight(m_ij)
        m_i = self.aggregate(m_ij, **aggr_kwargs)

        node_feats = self.node_norm(kwargs["x"], kwargs["batch"])
        node_out = self.node_mlp(torch.cat([node_feats, m_i], dim=-1))
        node_out = kwargs["x"] + node_out

        return self.update((node_out, coors_out), **update_kwargs)


class EGCL_block(nn.Module):
    def __init__(self, node_hidden_dim, edge_dim):
        super(EGCL_block, self).__init__()
        self.egcl = EGCL(node_hidden_dim, edge_dim)
        self.act = Softplus()
        self.lin = Linear(node_hidden_dim + 4 * 3, node_hidden_dim)
        self.layer_normal = nn.LayerNorm(node_hidden_dim + 4 * 3)
        self.time_emb = TimeEncoder(node_hidden_dim)
        self.silu = nn.SiLU()
        self.linear_t = nn.Linear(node_hidden_dim, node_hidden_dim)

    def forward(self, x, edge_index, edge_attr, edge_length, batch, time_step, pos):
        time_emb = self.linear_t(self.silu(self.time_emb(time_step, batch)))
        x = x + time_emb
        node_out, coors_out = self.egcl(x, edge_index, edge_attr, edge_length, batch, pos)
        coors_out = coors_out.reshape(-1, 4 * 3)
        output = torch.cat([node_out, coors_out], dim=-1)
        output = self.act(output)
        output = self.lin(self.layer_normal(output))
        return output


class EGNN(nn.Module):
    def __init__(self, node_input_dim=46, node_hidden_dim=128, edge_dim=8, node_output_dim=64, num_layer=4):
        super().__init__()
        self.lin_input = Linear(node_input_dim, node_hidden_dim)
        self.block_list = ModuleList()
        for _ in range(num_layer):
            block = EGCL_block(node_hidden_dim=node_hidden_dim, edge_dim=edge_dim)
            self.block_list.append(block)
        self.lin_output = Linear(node_hidden_dim, node_output_dim)

    def forward(self, x, edge_index, edge_attr, edge_length, batch, time_step, pos):
        hidden = self.lin_input(x)
        for block in self.block_list:
            hidden = hidden + block(hidden, edge_index, edge_attr, edge_length, batch, time_step, pos)
        hidden = self.lin_output(hidden)
        return hidden


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1e-2):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


# ==================== 新增：双路径结构编码器 ====================
class DualPathStructureEncoder(nn.Module):
    """
    双路径结构编码器：局部 EGNN + 全局 Transformer，动态融合
    """
    def __init__(self,
                 node_input_dim=46,
                 node_hidden_dim=128,
                 edge_dim=8,
                 node_output_dim=128,
                 num_layer=4,
                 global_hidden_dim=128,
                 global_n_head=4,
                 global_n_layers=2,
                 fusion='gate'):   # 'gate' or 'add' or 'concat'
        super().__init__()
        # 局部路径：原有 EGNN
        self.local_encoder = EGNN(
            node_input_dim=node_input_dim,
            node_hidden_dim=node_hidden_dim,
            edge_dim=edge_dim,
            node_output_dim=node_output_dim,
            num_layer=num_layer
        )
        # 全局路径：Transformer
        self.global_encoder = GlobalStructureEncoder(
            input_dim=node_output_dim,
            hidden_dim=global_hidden_dim,
            n_head=global_n_head,
            n_layers=global_n_layers
        )
        self.fusion = fusion
        if fusion == 'gate':
            self.gate = nn.Sequential(
                nn.Linear(node_output_dim * 2, node_output_dim),
                nn.Sigmoid()
            )
        elif fusion == 'concat':
            self.fusion_proj = nn.Linear(node_output_dim * 2, node_output_dim)

    def forward(self, x, edge_index, edge_attr, edge_length, batch, time_step, pos):
        # 局部路径输出
        h_local = self.local_encoder(x, edge_index, edge_attr, edge_length, batch, time_step, pos)
        # 全局路径：对每个样本分别处理
        batch_unique = batch.unique()
        h_global_list = []
        for b in batch_unique:
            mask = (batch == b)
            h_local_b = h_local[mask]   # (L_b, dim)
            h_global_b = self.global_encoder(h_local_b.unsqueeze(0)).squeeze(0)
            h_global_list.append(h_global_b)
        h_global = torch.cat(h_global_list, dim=0)

        # 融合
        if self.fusion == 'add':
            h_out = h_local + h_global
        elif self.fusion == 'gate':
            gate = self.gate(torch.cat([h_local, h_global], dim=-1))
            h_out = gate * h_local + (1 - gate) * h_global
        elif self.fusion == 'concat':
            h_out = self.fusion_proj(torch.cat([h_local, h_global], dim=-1))
        else:
            h_out = h_local

        return h_out