from train_mdnet_tensorboard import *
print(opts['model_path'])
opts['model_path'] = '../models/pymdnet-vot-attmask2-conv1&se3&loss-1011.pth'
print(opts['model_path'])
train_mdnet()