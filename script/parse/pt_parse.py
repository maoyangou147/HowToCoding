import torch

# 加载模型权重
weights = torch.load('/home/bob/experiment/dwz/mypt/yolov8x-seg.pt')

# 打印权重
for name, param in weights.items():
    print('Layer: ', name)
    print('Size: ', param.size())
    print('Values: ', param)