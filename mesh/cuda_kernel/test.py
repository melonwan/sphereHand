import torch
import depth_rasterization

vertices = torch.ones(1, 4, 4).cuda()
dms = depth_rasterization.forward(16, 16, vertices)
print(dms)
print(dms.shape)