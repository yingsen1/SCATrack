#coding:utf-8
import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.iteritems():
            if p is None: continue
            
            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k
            
            if name not in params:
                params[name] = p
            else:
                print('%s have already in param, so plus _2' %(name))
                name = name + '_' + '_2'
                params[name] = p
                # raise RuntimeError("Duplicated param name: %s" % (name))


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x**2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq,pad,pad,pad,pad),2),
                            torch.cat((pad,x_sq,pad,pad,pad),2),
                            torch.cat((pad,pad,x_sq,pad,pad),2),
                            torch.cat((pad,pad,pad,x_sq,pad),2),
                            torch.cat((pad,pad,pad,pad,x_sq),2)),1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:,2:-2,:,:]
        x = x / ((2.+0.0001*x_sumsq)**0.75)
        return x


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K
        self.layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(),
                                        LRN(),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU())),
                ('fc4',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU())),
                ('fc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU()))]))

        self.seblocks = nn.Sequential(OrderedDict([
            ('conv3_se', nn.Sequential(nn.AvgPool2d(3, stride=1),
                nn.Linear(in_features=512, out_features=32),
                nn.ReLU(),
                nn.Linear(in_features=32, out_features=512),
                nn.Sigmoid()
                ))]))

       
        self.attmask = nn.Sequential(OrderedDict([
            ('conv1_theta', nn.Sequential(nn.Conv2d(96, 48, kernel_size=1, stride=1))),
            ('conv1_phi', nn.Sequential(nn.Conv2d(96, 48, kernel_size=1, stride=1))),
            ('conv1_g', nn.Sequential(nn.Conv2d(96, 48, kernel_size=1, stride=1))),
            ('conv1_out', nn.Sequential(nn.Conv2d(48, 96, kernel_size=1, stride=1),
                nn.BatchNorm2d(96)))
            ]))
        nn.init.constant(self.attmask[3][0].weight, 0)
        nn.init.constant(self.attmask[3][0].bias, 0)


        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5), 
                                                     nn.Linear(512, 2)) for _ in range(K)])
        
        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches):
            append_params(self.params, module, 'fc6_%d'%(k))
        for name, module in self.seblocks.named_children():
            append_params(self.params, module, name)
        for name, module in self.attmask.named_children():
            append_params(self.params, module, name)

    def set_learnable_params(self, layers):
        for k, p in self.params.iteritems():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False
 
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        return params
    
    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6', mode='one'):
        # mode = one\all
        run = False # the signal to control the forward

        # conv1->fc5中运行16
        # for name, module in self.layers.named_children():
        #     if name == in_layer: # 判断in_layer是否在模型里边，从而控制forward
        #         run = True
        #     if run:
        #         x = module(x)
        #         if name == 'conv3': #更改conv3的形状，相当于flatten()
        #             x = x.view(x.size(0), -1) # size()[0] is Nbatch?
        #         if name == out_layer:
        #             return x

        # 对conv1->fc5的改写  # TODO:debug一下这个in_layer
        for name, module in self.layers.named_children():# {conv1, conv2,...,}迭代器
            if name == in_layer:
                run = True
            if run:
                if name == 'conv1':
                    x = module(x)

                    mask_x = x
                    # spatial-process                    
                    batch_size = mask_x.size(0)
                    mask_x_theta = self.attmask[0](mask_x)
                    mask_x_phi = self.attmask[1](mask_x)
                    mask_x_g = self.attmask[2](mask_x)
                    mask_x_theta = mask_x_theta.view(batch_size, 48, 625).permute(0, 2, 1)
                    mask_x_phi = mask_x_phi.view(batch_size, 48, 625)
                    mask_x_theta = torch.bmm(mask_x_theta, mask_x_phi)
                    mask_x_theta = F.softmax(mask_x_theta, -1) #??
                    mask_x_g = mask_x_g.view(batch_size, 48, 625).permute(0, 2, 1)
                    mask_x_theta = torch.bmm(mask_x_theta, mask_x_g)
                    mask_x_theta = mask_x_theta.permute(0, 2, 1).contiguous()
                    mask_x_theta = mask_x_theta.view(batch_size, 48, 25, 25)
                    mask_x_theta = self.attmask[3](mask_x_theta)
                    mask_x = mask_x_theta + mask_x
                    x = mask_x
                   
                elif name == 'conv2':
                    x = module(x)
                  
                elif name == 'conv3':
                    x = module[0](x) # conv 
                    # 这里为seblock运算
                    original_x = x # 新分支
                                        # channel-process
                    x = self.seblocks[0][0](x) # GAP
                    x = x.view(x.size(0), -1) # flatten
                    x = self.seblocks[0][1](x) # linear
                    x = self.seblocks[0][2](x) # RELU
                    x = self.seblocks[0][3](x) # linear
                    x = self.seblocks[0][4](x) # Sigmoid
                    x = x.view(x.size(0), x.size(1), 1, 1)
                    
                    x = original_x * x
                    # channel & spatial attention
                    # x = x * original_x # channel-wise

                    # x = 0.5 * (original_x * mask_x) + 0.5*x# spatial attention
                    

                    # activation 
                    x = module[1](x) # ReLU
                    x = x.view(x.size(0), -1)

                elif name == 'fc4':
                    x = module(x)
                elif name == 'fc5':
                    x = module(x)

                if name == out_layer:
                    return x
        if mode == 'one':
            x = self.branches[k](x)
            if out_layer == 'fc6':
                return x
            elif out_layer == 'fc6_softmax':
                return F.softmax(x)
        elif mode == 'all':
            for j in range(self.K):
                p_score = self.branches[j](x)
                if j == k:
                    k_score = p_score.clone()
                if j == 0:
                    p_score_all = p_score[:,1:].clone()
                else:
                    p_score_all = torch.cat((p_score_all, p_score[:, 1:].clone()), 1)
            if out_layer == 'fc6':
                return p_score_all, k_score

    
    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers_0 = states['shared_layers_conv&fc']
        shared_layers_1 = states['shared_layers_seblock']
        shared_layers_2 = states['shared_layers_attmask']
        self.layers.load_state_dict(shared_layers_0)
        self.seblocks.load_state_dict(shared_layers_1)
        self.attmask.load_state_dict(shared_layers_2)
    
    def load_mat_model(self, matfile):
        print(matfile)
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]
        
        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i*4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3,2,0,1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:,0])


##################################
    

class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()
 
    def forward(self, pos_score, neg_score):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:,1]   ## add dim=1, so do next line
        neg_loss = -F.log_softmax(neg_score, dim=1)[:,0]
        
        # loss = (pos_loss.sum() + neg_loss.sum()) / (len(pos_loss)+len(neg_loss))
        loss = (pos_loss.sum() + neg_loss.sum())
        return loss

class InstanceLoss(nn.Module):
    def __init__(self):
        super(InstanceLoss, self).__init__()

    def forward(self, p_score_all, k, num):
        k_list_0 = np.random.permutation(p_score_all.size(1)) # ramdomly make K index
        k_index = np.where(k_list_0 == k) # k's index
        # k_list_0[[0, k_index]] = k_list_0[[k_index, 0]] # swap k to last location
        tmp = k_list_0[0]
        k_list_0[0] = k_list_0[k_index]
        k_list_0[k_index] = tmp
        k_list_selected = k_list_0[:num] # selected K index
        loss = -F.log_softmax(p_score_all[:, k_list_selected], dim=1)[:, 0] # k from 0
        # loss = loss.sum() / len(loss) # all pos_regions loss
        loss = loss.sum()
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        
        pos_correct = (pos_score[:,1] > pos_score[:,0]).sum().float()
        neg_correct = (neg_score[:,1] < neg_score[:,0]).sum().float()
        
        pos_acc = pos_correct / (pos_score.size(0) + 1e-8)
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0]


class Precision():
    def __call__(self, pos_score, neg_score):
        
        scores = torch.cat((pos_score[:,1], neg_score[:,1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0)+1e-8)
        
        return prec.data[0]
