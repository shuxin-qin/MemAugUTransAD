import torch
import torch.nn as nn

from modules import ConvLayer, MaxPoolLayer
from transformer import EncoderLayer, DecoderLayer
from memory import MemoryLocal

'''
----------------------------------------------------------------------------
--------------------------------------------------------------- layer num = 3
----------------------------------------------------------------------------
results, py3, with mem  win=100  epcho=20: head=4 mem_slot=64
(F1, P, R)
--msl: (95.53,95.63,95.43)--20220722_105825
--smap:(96.94,96.63,97.26)--20220722_153850
--swat:(88.17,96.13,81.43)--20220726_094113
--psm: (91.91,95.49,88.58)--20220722_110753
---------------------------------------------------------------------------
mem_slot=32
--msl: (93.30,91.23,95.43)--20220728_111631
--smap:(93.21,96.75,89.91)--20220729_153720
--swat:(85.47,84.73,86.22)--20220728_124226
--psm: (89.40,90.19,88.63)--20220728_135315
mem_slot=128
--msl: (94.11,96.12,92.18)--20220728_143030
--smap:(96.42,95.66,97.20)--20220728_144248
--swat:(87.52,95.52,80.75)--20220728_150534
--psm: (92.44,96.71,88.54)--20220728_154902
mem_slot=256
--msl: (94.52,94.91,94.13)--20220728_171236
--smap:(96.31,95.66,97.26)--20220729_092456
--swat:(86.66,95.97,79.01)--20220729_123011
--psm: (92.09,95.87,88.59)--20220728_160131
----------------------------------------------------------------------------
results, py3, with mem (without projection loss)  win=100  epcho=20: head=4 mem_slot=64
(F1, P, R)
--msl: (94.97,94.52,95.43)--20220728_095515
--smap:(92.17,94.29,90.13)--20220728_105004
--swat:(88.10,95.04,82.10)--20220727_183554
--psm: (90.10,89.81,90.40)--20220727_165615
----------------------------------------------------------------------------
results, py3, without mem  win=100  epcho=20:  head=4 mem_slot=64
(F1, P, R)
--msl: (93.51,93.53,93.48)--20220727_154954
--smap:(92.26,94.70,89.95)--20220727_152943
--swat:(84.62,90.98,79.10)--20220727_150907
--psm: (89.84,95.13,85.11)--20220727_141345
----------------------------------------------------------------------------
results, py3, without pyramid  win=100  epcho=20:  head=4 mem_slot=64
(F1, P, R)
--msl: (91.97,90.71,93.27)--20220731_105216
--smap:(91.44,93.02,89.92)--20220731_110824
--swat:(82.95,98.78,71.49)--20220731_121653
--psm: (85.73,81.51,90.40)--20220731_170744

-----------------------------------------------------------------------------
--------------------------------------------------------------- layer num = 2
-----------------------------------------------------------------------------
results, py2, with mem  win=100  epcho=20: head=4 mem_slot=64
(F1, P, R)
--msl: (95.68,97.30,94.13)--20220726_171850
--smap:(91.23,92.35,90.13)--20220726_172900
--swat:(85.49,92.23,79.66)--20220726_164302
--psm: (90.68,91.06,90.29)--20220726_183213

results, py2, without mem  win=100  epcho=20: head=4 mem_slot=64
(F1, P, R)
--msl: (95.46,96.15,94.78)--20220727_124317
--smap:(90.89,85.36,97.20)--20220727_125213
--swat:(84.14,98.05,73.69)--20220727_131237
--psm: (86.62,84.90,88.41)--20220727_134022

------------------------------------------------------------------------------
---------------------------------------------------------------- layer num = 1
------------------------------------------------------------------------------
results, py1, with mem  win=100  epcho=20: head=4 mem_slot=64
(F1, P, R)
--msl: (90.50,93.09,88.06)--20220727_123547
--smap:(90.10,95.64,85.18)--20220727_093615
--swat:(83.80,97.01,73.76)--20220727_095055
--psm: (84.53,99.24,73.61)--20220727_102130

results, py1, without mem  win=100  epcho=20: head=4 mem_slot=64
(F1, P, R)
--msl: (90.22,91.42,89.05)--20220727_113903
--smap:(88.72,87.62,89.85)--20220727_111839
--swat:(82.29,98.50,70.67)--20220727_105611
--psm: (84.43,82.39,86.58)--20220727_101053

'''


class PYRAMID_TRANS_MEM(nn.Module):
    """ EMB_2TRANS model class.
    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        dropout=0.2
    ):
        super(PYRAMID_TRANS_MEM, self).__init__()
        self.window_size = window_size
        self.out_dim = out_dim

        if n_features <= 32:
            self.f_dim = 32
        elif n_features <= 64:
            self.f_dim = 64
        elif n_features <= 128:
            self.f_dim = 128


        self.conv = ConvLayer(n_features, kernel_size)
        self.conv_dim = nn.Linear(n_features, self.f_dim)

        self.fuse_type = 0    # 0:同时使用pyramid and multi-resolution  1:仅使用pyramid  2:仅使用multi-resolution
        self.layer_num = 3    # 1:一层  2:两层 3:三层
        self.use_mem = 1      # 0:不使用   1:使用local memory module
        self.use_pyramid = 1  # 0:不进行下采用和上采样   1:使用下采样和上采样
        self.num_slots = 64   # 32 64 128 256

        heads = 4
        self.win2 = int(self.window_size/2)
        self.win3 = int(self.win2/2)
        self.win4 = int(self.win3/2 + 0.5)

        if self.layer_num >= 1:
            self.enc_layer1 = EncoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_mem == 1:
                self.mem1 = MemoryLocal(num_slots=self.num_slots, slot_dim=self.f_dim)
            self.dec_layer1 = DecoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)

        if self.layer_num >= 2:
            self.enc_layer2 = EncoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_mem == 1:
                self.mem2 = MemoryLocal(num_slots=self.num_slots, slot_dim=self.f_dim)
            self.dec_layer2 = DecoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_pyramid == 1:
                self.d_sampling2 = MaxPoolLayer()
                self.u_sampling2 = nn.Conv1d(in_channels=self.win2, out_channels=self.window_size, kernel_size=3, padding=1)

        if self.layer_num >= 3:
            self.enc_layer3 = EncoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_mem == 1:
                self.mem3 = MemoryLocal(num_slots=self.num_slots, slot_dim=self.f_dim)
            self.dec_layer3 = DecoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_pyramid == 1:
                self.d_sampling3 = MaxPoolLayer()
                self.u_sampling3 = nn.Conv1d(in_channels=self.win3, out_channels=self.win2, kernel_size=3, padding=1)

        if self.layer_num == 4:
            self.enc_layer4 = EncoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_mem == 1:
                self.mem4 = MemoryLocal(num_slots=self.num_slots, slot_dim=self.f_dim)
            self.dec_layer4 = DecoderLayer(n_feature=self.f_dim, num_heads=heads, hid_dim=self.f_dim, dropout=dropout)
            if self.use_pyramid == 1:
                self.d_sampling4 = MaxPoolLayer()
                self.u_sampling4 = nn.Conv1d(in_channels=self.win4, out_channels=self.win3, kernel_size=3, padding=1)

        # mlp layer to resconstruct output
        self.mlp = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(),
            nn.Linear(self.f_dim, self.out_dim)
        )


    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # print('x:', x.shape)
        x = self.conv(x)
        x = self.conv_dim(x)

        if self.layer_num == 1:
            enc1, _ = self.enc_layer1(x)
            if self.use_mem == 1:
                mem1, weight1 = self.mem1(enc1)
                dec1, _, _ = self.dec_layer1(x, mem1)
                # memory loss
                loss_mem = weight1
            else:
                dec1, _, _ = self.dec_layer1(x, enc1)

        elif self.layer_num == 2:
            enc1, _ = self.enc_layer1(x)
            if self.use_pyramid == 1:
                down2 = self.d_sampling2(enc1)
                enc2, _ = self.enc_layer2(down2)
            else:
                enc2, _ = self.enc_layer2(enc1)
            if self.use_mem == 1:
                mem1, weight1 = self.mem1(enc1)
                mem2, weight2 = self.mem2(enc2)
                if self.use_pyramid == 1:
                    dec2, _, _ = self.dec_layer2(down2, mem2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, mem1)
                else:
                    dec2, _, _ = self.dec_layer2(enc1, mem2)
                    dec1, _, _ = self.dec_layer1(dec2, mem1)
                # memory loss
                loss_mem = weight1 + weight2
            else:
                if self.use_pyramid == 1:
                    dec2, _, _ = self.dec_layer2(down2, enc2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, enc1)
                else:
                    dec2, _, _ = self.dec_layer2(enc1, enc2)
                    dec1, _, _ = self.dec_layer1(dec2, enc1)

        elif self.layer_num == 3:
            enc1, _ = self.enc_layer1(x)
            if self.use_pyramid == 1:
                down2 = self.d_sampling2(enc1)
                enc2, _ = self.enc_layer2(down2)
                down3 = self.d_sampling3(enc2)
                enc3, _ = self.enc_layer3(down3)
            else:
                enc2, _ = self.enc_layer2(enc1)
                enc3, _ = self.enc_layer3(enc2)
            if self.use_mem == 1:
                mem1, weight1 = self.mem1(enc1)
                mem2, weight2 = self.mem2(enc2)
                mem3, weight3 = self.mem3(enc3)
                if self.use_pyramid == 1:
                    dec3, _, _ = self.dec_layer3(down3, mem3)
                    up3 = self.u_sampling3(dec3)
                    dec2, _, _ = self.dec_layer2(up3, mem2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, mem1)
                else:
                    dec3, _, _ = self.dec_layer3(enc2, mem3)
                    dec2, _, _ = self.dec_layer2(dec3, mem2)
                    dec1, _, _ = self.dec_layer1(dec2, mem1)
                # memory loss
                loss_mem = weight1 + weight2 + weight3
            else:
                if self.use_pyramid == 1:
                    dec3, _, _ = self.dec_layer3(down3, enc3)
                    up3 = self.u_sampling3(dec3)
                    dec2, _, _ = self.dec_layer2(up3, enc2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, enc1)
                else:
                    dec3, _, _ = self.dec_layer3(enc2, enc3)
                    dec2, _, _ = self.dec_layer2(dec3, enc2)
                    dec1, _, _ = self.dec_layer1(dec2, enc1)

        elif self.layer_num == 4:
            enc1, _ = self.enc_layer1(x)
            if self.use_pyramid == 1:
                down2 = self.d_sampling2(enc1)
                enc2, _ = self.enc_layer2(down2)
                down3 = self.d_sampling3(enc2)
                enc3, _ = self.enc_layer3(down3)
                down4 = self.d_sampling3(enc3)
                enc4, _ = self.enc_layer3(down4)
            else:
                enc2, _ = self.enc_layer2(enc1)
                enc3, _ = self.enc_layer3(enc2)
                enc4, _ = self.enc_layer3(enc3)
            if self.use_mem == 1:
                mem1, weight1 = self.mem1(enc1)
                mem2, weight2 = self.mem2(enc2)
                mem3, weight3 = self.mem3(enc3)
                mem4, weight4 = self.mem3(enc4)
                if self.use_pyramid == 1:
                    dec4, _, _ = self.dec_layer3(down4, mem4)
                    up4 = self.u_sampling3(dec4)
                    dec3, _, _ = self.dec_layer3(up4, mem3)
                    up3 = self.u_sampling3(dec3)
                    dec2, _, _ = self.dec_layer2(up3, mem2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, mem1)
                else:
                    dec4, _, _ = self.dec_layer3(enc3, mem4)
                    dec3, _, _ = self.dec_layer3(dec4, mem3)
                    dec2, _, _ = self.dec_layer2(dec3, mem2)
                    dec1, _, _ = self.dec_layer1(dec2, mem1)
                # memory loss
                loss_mem = weight1 + weight2 + weight3
            else:
                if self.use_pyramid == 1:
                    dec4, _, _ = self.dec_layer3(down4, enc4)
                    up4 = self.u_sampling3(dec4)
                    dec3, _, _ = self.dec_layer3(up4, enc3)
                    up3 = self.u_sampling3(dec3)
                    dec2, _, _ = self.dec_layer2(up3, enc2)
                    up2 = self.u_sampling2(dec2)
                    dec1, _, _ = self.dec_layer1(up2, enc1)
                else:
                    dec4, _, _ = self.dec_layer3(enc3, enc4)
                    dec3, _, _ = self.dec_layer3(dec4, enc3)
                    dec2, _, _ = self.dec_layer2(dec3, enc2)
                    dec1, _, _ = self.dec_layer1(dec2, enc1)

        recon = self.mlp(dec1)

        if self.use_mem == 0:
            loss_mem = torch.zeros([1]).cuda()

        return recon, loss_mem

