# 模型训练之后查看模型参数量

import torch
from model import Model


model = Model()
state_dict = torch.load()
model.load_state_dict(state_dict)

# 计算参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"模型参数量为:{total_params}")
