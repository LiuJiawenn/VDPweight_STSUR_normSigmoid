import torch
import torch.nn as nn
import torch.nn.functional as f
from model.spq_subnet import SPQSubNet
from model.tpq_subnet import TPQSubNet

class VDPWeightSTSUR(nn.Module):
    def __init__(self, patch_per_frame=64, key_frame_nb=12):
        super(VDPWeightSTSUR, self).__init__()
        self.patch_per_frame = patch_per_frame
        self.key_frame_nb = key_frame_nb
        self.patches = self.patch_per_frame * self.key_frame_nb

        self.spq_subnet = SPQSubNet()
        self.tpq_subnet = TPQSubNet()
        self.vdp_subnet = SPQSubNet()

        self.sc_fc1 = nn.Linear(512 * 3, 512)
        self.sc_fc2 = nn.Linear(512, 1)

        self.o_fc1 = nn.Linear(512, 512)
        self.o_fc2 = nn.Linear(512, 1)

        self.vdp_fc1 = nn.Linear(512, 512)
        self.vdp_fc2 = nn.Linear(512, 1)

    def forward(self, x):
        hs = self.spq_subnet(x[0].float())
        hc = self.spq_subnet(x[1].float())
        ho = self.tpq_subnet(x[2].float())
        hv = self.vdp_subnet(x[3].float())

        # 连接空间特征
        hsc = torch.cat((hc-hs, hc, hs), dim=1)
        hsc = hsc.view(self.patches, 512*3)
        hsc = f.dropout(f.relu(self.sc_fc1(hsc)), p=0.5)
        spq = self.sc_fc2(hsc)

        # 时间特征
        ho = ho.view(self.patches, 512)
        ho = f.dropout(f.relu(self.o_fc1(ho)), p=0.5)
        tpq = self.o_fc2(ho)

        # 时空特征融合
        stpq = torch.mul(spq, tpq)  # 每个patch的时空特征评分，应该是0-1之间的，代表每个patch的SUR
        maxpq = stpq.max()
        stpq = (stpq / maxpq) * 12 - 6
        stpsur = f.sigmoid(stpq)  # 使用sigmoid将每个patch的分转换到0-1区间

        # minpq = stpq.min()
        # stpq = stpq-minpq
        # maxpq = stpq.max()
        # stpq = (stpq / maxpq) * 12 - 6
        # stpsur = f.sigmoid(stpq)  # 使用sigmoid将每个patch的分转换到0-1区间

        # VDP权重
        hv = hv.view(self.patches, 512)
        hv = f.dropout(f.relu(self.vdp_fc1(hv)), p=0.5)
        hv = self.vdp_fc2(hv)
        hv = hv-torch.min(hv)
        hv_flat = hv.view(self.key_frame_nb, self.patch_per_frame)
        norm = hv_flat.sum(dim=1, keepdim=True) + 1e-10
        vdp_weight = hv_flat/norm

        # 评分与权重对应元素相乘，再按行求和（每行是一帧的所有patch）
        frame_sur = torch.sum(torch.mul(stpsur.reshape(self.key_frame_nb, self.patch_per_frame), vdp_weight), dim=1)
        outputs = torch.mean(frame_sur)

        return outputs