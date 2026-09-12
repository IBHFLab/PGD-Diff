import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from modules.sequence.encode import index_to_onehot
from util.embed.sequence import fasta_to_index


class pgddiffDataset(Dataset):
    def __init__(self, ACP_id_list, ACP_fasta_list, ACP_strut_data, nonACP_id_list, nonACP_fasta_list,
                 nonACP_strut_data):
        self.ACP_id_list = ACP_id_list
        self.ACP_fasta_list = ACP_fasta_list
        self.ACP_strut_data = ACP_strut_data

        self.nonACP_id_list = nonACP_id_list
        self.nonACP_fasta_list = nonACP_fasta_list
        self.nonACP_strut_data = nonACP_strut_data

    def __getitem__(self, index):
        ACP_id = self.ACP_id_list[index]
        ACP_fasta = self.ACP_fasta_list[index]
        ACP_pos = torch.tensor(self.ACP_strut_data[ACP_id])

        nonACP_id = self.nonACP_id_list[index]
        nonACP_fasta = self.nonACP_fasta_list[index]
        nonACP_pos = torch.tensor(self.nonACP_strut_data[nonACP_id])

        ACP_logit = index_to_onehot(fasta_to_index(ACP_fasta))
        nonACP_logit = index_to_onehot(fasta_to_index(nonACP_fasta))

        data = Data(x=index, pos=ACP_pos, fasta=ACP_fasta, logit=ACP_logit, nonacp_pos=nonACP_pos,
                    nonacp_fasta=nonACP_fasta, nonacp_logit=nonACP_logit)
        return data

    def __len__(self):
        return len(self.ACP_fasta_list)
