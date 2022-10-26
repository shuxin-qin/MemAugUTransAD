import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryLocal(nn.Module):
    def __init__(self, num_slots, slot_dim):
        super(MemoryLocal, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.memMatrix = nn.Parameter(torch.empty(num_slots, slot_dim))  # M,C
        self.criterion = nn.MSELoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        :param x: query features with size [N,C], where N is the number of query items,
                  C is same as dimension of memory slot

        :return: query output retrieved from memory, with the same size as x.
        """
        # flatten
        b, t, c = x.shape
        query = x.contiguous().view(b*t, c)  # [b*t, c]

        # cosine similarity for feature retrieve and memory update
        # m_norm = self.memMatrix / torch.norm(self.memMatrix, dim=-1, keepdim=True) # 方差归一化，即除以各自的模
        # q_norm = query / torch.norm(query, dim=-1, keepdim=True) # 方差归一化，即除以各自的模
        # dist = torch.matmul(q_norm, m_norm.T)   # [b*t, c] * [c, n] == [b*t, n]  计算query 和 memory之间的 cosine similarity
        dist = torch.matmul(query, self.memMatrix.T)   # [b*t, c] * [c, n] == [b*t, n]  计算query 和 memory之间的 cosine similarity

        # softmax score
        m_score = F.softmax(dist, dim=1)   # [b*t, n] 通过softmax进行归一化得到距离得分矩阵

        # read memory
        select_mem = torch.matmul(m_score, self.memMatrix) # [b*t, n] * [n, c] == [b*t, c]

        feat = select_mem.view(b, t, c)  # 还原到原来的维度 [b, t, c]

        # calculate memory sparsity loss
        s_loss = torch.mean(torch.sum(-m_score * torch.log(m_score + 1e-12), dim=1))

        # projection loss
        p_loss = self.criterion(query, select_mem)

        loss = s_loss + p_loss

        return feat, loss