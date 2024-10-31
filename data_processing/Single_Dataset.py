from torch.utils.data import Dataset
import numpy as np
import torch
import os

class Single_Dataset(Dataset):

    #file_list: 데이터셋으로 사용할 파일 경로의 리스트(.npz 파일 경로 포함)
    def __init__(self,file_list):
        self.listfiles=file_list

    #idx: 인덱스 값(데이터셋 내 특정 위치의 파일을 참조)
    def __getitem__(self, idx):
        file_path=self.listfiles[idx]
        data=np.load(file_path)
        # H=data['H']
        # A1=data['A1']
        # A2 = data['A2']
        # Y=data['Y']
        # V=data['V']
        # H = torch.from_numpy(H).float()
        # A1 = torch.from_numpy(A1).float()
        # A2 = torch.from_numpy(A2).float()
        # Y = torch.from_numpy(Y).float()
        # V = torch.from_numpy(V).float()
        return data


    #데이터셋의 총 길이를 반환
    def __len__(self):
        return len(self.listfiles)
