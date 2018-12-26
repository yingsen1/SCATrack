from model import *

import torch
import tensorboardX
import torch.onnx


input = torch.rand(1, 3, 107, 107)
input = torch.autograd.Variable(input)
model = MDNet()
proto=torch.onnx.export(model, input, "mdnet.proto", verbose=True)
writer=tensorboardX.SummaryWriter("./logs/")
writer.add_graph_onnx("./mdnet.proto")