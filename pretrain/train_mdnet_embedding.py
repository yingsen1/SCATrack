import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

from data_prov import *
from model import *
from options import *
from tensorboardX import SummaryWriter
import json

img_home = '../../tracking_benchmark/dataset/'
data_path = 'data/vot-otb.pkl'
VID_data_path = 'data/ilsvrc_train_500.json'
VID_home = '../../tracking_benchmark/ILSVRC2015/Data/VID/train/'

def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


def train_mdnet():
    # init tools
    writer = SummaryWriter()



    # ## Init dataset ## for VOT
    # with open(data_path, 'rb') as fp:
    #     data = pickle.load(fp)

    # K = len(data)
    # dataset = [None]*K
    # for k, (seqname, seq) in enumerate(data.iteritems()):
    #     img_list = seq['images']
    #     gt = seq['gt']
    #     img_dir = os.path.join(img_home, seqname)
    #     dataset[k] = RegionDataset(img_dir, img_list, gt, opts)

    ## Init dataset ## for VID
    with open(VID_data_path, 'r') as f:
        data = json.load(f)

    K = len(data)
    dataset = [None]*K
    for k, seq in enumerate(iter(data)):
        seq_name = seq['seq_name']
        start_frame = seq['start_frame']
        end_frame = seq['end_frame']
        gt = np.array(seq['gt'])
        im_width = seq['im_width']
        im_height = seq['im_height']
        img_dir = os.path.join(VID_home, seq_name)
        dataset[k] = ILSVRC2015RegionDataset(img_dir, start_frame, end_frame, gt, im_width, im_height, opts)


    ## Init model ##
    print(opts['init_model_path'])
    print(K)
    model = MDNet(opts['init_model_path'], K)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])
        
    ## Init criterion and optimizer ##
    criterion_cls = BinaryLoss()
    criterion_ins = InstanceLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'])

    best_prec = 0.
    for i in range(opts['n_cycles']):
        print "==== Start Cycle %d ====" % (i)
        k_list = np.random.permutation(K)
        t_start = time.time()
        prec = np.zeros(K)
        # loss = 0
        loss_list = [None]*K
        loss_print = 0
        # model.zero_grad() # eliminate grad 
        for j,k in enumerate(k_list):
            tic = time.time()
            pos_regions, neg_regions = dataset[k].next()
            
            pos_regions = Variable(pos_regions)
            neg_regions = Variable(neg_regions)
        
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()

            p_score_all, pos_score = model(pos_regions, k, mode='all') ##zhuyi
      
            neg_score = model(neg_regions, k)

            # loss = criterion(pos_score, neg_score)
            loss_cls = criterion_cls(pos_score, neg_score)
            loss_ins = criterion_ins(p_score_all, k, opts['ins_loss_num'])
            loss = loss_cls + loss_ins * opts['alpha']
            loss_print += loss.data[0]
            loss_list[k] = loss_cls.data[0]+loss_ins.data[0]*opts['alpha']
            # ===
            # if j % opts[] == 0:
            #     model.zero_grad()
            #     loss.backward()
            #     l = k
            #     prec[k] = evaluator(pos_score, neg_score)


            
            # elif j % opts['accu_grad_every'] == opts['accu_grad_every']-1:
            #     # model.zero_grad()
            #     loss.backward() # accumlate grad
            #     torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])  # TODO:anyquestion
            #     optimizer.step()
            #     prec[k] = evaluator(pos_score, neg_score)
            #     toc = time.time()-tic
            #     print "Cycle %2d, K %2d (%2d, %2d), Loss (%.3f, %.3f), Prec (%.3f, %.3f), Time %.3f, L_cls %.3f, L_ins %.3f" % \
            #         (i, j, l, k, loss_list[l] ,loss_list[k], prec[l], prec[k], toc, loss_cls.data[0], loss_ins.data[0])
            # # writer.add_scalar('branch/'+str(k), loss_cls.data[0]+loss_ins.data[0]*opts['alpha'], i) # mark each branch loss
            # ===
            if j % opts['accu_grad_every'] == opts['accu_grad_every']-1:
                # model.zero_grad()
                loss.backward() # accumlate grad
                torch.nn.utils.clip_grad_norm(model.parameters(), opts['grad_clip'])  # TODO:anyquestion
                optimizer.step()
                prec[k] = evaluator(pos_score, neg_score)
                toc = time.time()-tic
                print "Cycle:%2d,K:%2d(%2d),Loss:(%.3f),Prec:(%.3f),Time:%.3f,L_cls:%.3f,L_ins:%.3f" % \
                    (i, j,  k, loss_list[k], prec[k], toc, loss_cls.data[0], loss_ins.data[0]*opts['alpha'])
            
            elif j % opts['accu_grad_every'] == 0:
                model.zero_grad()
                loss.backward()
                ## make a list to store previous k
                prec[k] = evaluator(pos_score, neg_score)
            else:
                loss.backward()
                prec[k] = evaluator(pos_score, neg_score)

        
        t_cycle = time.time() - t_start
        cur_prec = prec.mean()
        print "Cycle %2d,  Loss %.3f, Mean-Prec %.3f, Time %.3f" % \
                (i, loss_print, cur_prec, t_cycle)
        if i % 5 == 0: # mark every 10 epoch
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram(tag, value.clone().cpu().data.numpy(), i)
                # writer.add_histogram(tag+'/grad', value.grad.clone().cpu().data.numpy(), i)
            writer.add_scalar('data/cur_prec', cur_prec, i) # print precision

        # print "Mean Precision: %.3f" % (cur_prec)
        if cur_prec > best_prec:
            best_prec = cur_prec
            if opts['use_gpu']:
                model = model.cpu()
            states = {'shared_layers_conv&fc': model.layers.state_dict(),'shared_layers_seblock':model.seblocks.state_dict(),
            'shared_layers_attmask': model.attmask.state_dict(),'branchs':model.branches.state_dict()}
            print "Save model to %s" % opts['model_path']
            torch.save(states, opts['model_path'])
            if opts['use_gpu']:
                model = model.cuda()
    writer.export_scalars_to_json("./test.json")
    writer.close()

if __name__ == "__main__":
    train_mdnet()

