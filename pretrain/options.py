from collections import OrderedDict

opts = OrderedDict()
opts['use_gpu'] = True

opts['init_model_path'] = '../models/imagenet-vgg-m.mat'
opts['model_path'] = '../models/pymdnet-vot-seblock-r16-0921.pth'

opts['batch_frames'] = 8
opts['batch_pos'] = 32 # 32
opts['batch_neg'] = 96 # 96

opts['overlap_pos'] = [0.7, 1]
opts['overlap_neg'] = [0, 0.5]

opts['img_size'] = 107
opts['padding'] = 16

opts['lr'] = 0.0001
opts['w_decay'] = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10
opts['ft_layers'] = ['conv','fc']
opts['lr_mult'] = {'fc':10} 
opts['n_cycles'] = 100  # 50

# ins_loss
opts['alpha'] = 0.15
opts['ins_loss_num'] = 20
opts['accu_grad_every'] = 10