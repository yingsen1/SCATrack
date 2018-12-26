import sys
sys.path.insert(0,'../modules')
from model import *
model_path = '../models/pymdnet-vot-attmask-sig-0920.pth'
net = MDNet(model_path)