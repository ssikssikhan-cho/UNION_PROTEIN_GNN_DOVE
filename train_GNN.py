import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from multiprocessing import Pool
from model.layers import GAT_gate
import os
from ops.os_operation import mkdir
import shutil
import  numpy as np
from data_processing.Prepare_Input import Prepare_Input
from model.GNN_Model import GNN_Model
import torch
from ops.train_utils import count_parameters,initialize_model
from data_processing.collate_fn import collate_fn
from data_processing.Single_Dataset import Single_Dataset
from torch.utils.data import DataLoader
from ops.argparser import argparser
import os
import time
from prepare_learning_data import AverageMeter
from ops.argparser import argparser
from model.GNN_Model import GNN_Model
import torch
import os
from prepare_learning_data import collate_fn
from ops.train_utils import count_parameters,initialize_model
from data_processing.Single_Dataset import Single_Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model.GNN_Model import GNN_Model
import torch
from ops.train_utils import count_parameters,initialize_model
from data_processing.Single_Dataset import Single_Dataset
from torch.utils.data import DataLoader
from ops.argparser import argparser
import time



def train_GNN(model,train_dataloader,optimizer,loss_fn,device):
    
    
    #학습할 GNN 모델
    model.train()

    #Loss, Accu1 : 배치별 평균 손실과 정확도를 추적
    Loss = AverageMeter()
    Accu1 = AverageMeter()
    end_time = time.time()

    #배치 단위로 데이터를 가져옴
    for batch_idx, sample in enumerate(train_dataloader):
        # start = time.time()
        b = time.time()
        H, A1, A2, V, Atom_count, Y = sample
        batch_size = H.size(0)
        H, A1, A2, Y, V = H.cuda(), A1.cuda(), A2.cuda(), Y.cuda(), V.cuda()
        # end = time.time()
        # print("loop end, train start: ", end - start)

        #델의 예측을 수행
        pred = model.train_model((H, A1, A2, V, Atom_count), device)
        # start = time.time()
        # print("end train: ", start - end)

        #손실을 계산
        loss = loss_fn(pred, Y)
        # end = time.time()
        # print("loss end: ", end - start)

        #이전 단계의 기울기를 초기화
        optimizer.zero_grad()
        # start = time.time()
        # print("optimizer end:", start - end)
        #torch.nn.utils.clip_grad_norm_(model.parameters(), params['clip'])
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value = 1.0)
        #loss_list.append(loss)


        # 역전파를 통해 기울기를 계산
        loss.backward()

        #가중치 업데이트
        optimizer.step()


        #예측값 기반 정확도 업데이트
        Accu1.update(pred, batch_size)

        #배치 손실 업데이트
        Loss.update(loss.item(), batch_size)
        # end = time.time()

        # print("batch end: ",batch_idx, time.time() - b)


    return Loss.avg, Accu1.avg#, loss_list