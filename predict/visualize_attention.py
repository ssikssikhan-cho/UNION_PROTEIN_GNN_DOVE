#GNN 모델의 주의(attention) 가중치를 시각화
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
from predict.predict_single_input import init_model

#데이터 로더에서 배치 단위로 데이터를 가져와 주의 가중치를 계산하는 함수
def Get_Attention(dataloader,device,model):
    Final_atten1 = []
    Final_atten2 = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader):
            H, A1, A2, V, Atom_count = sample
            batch_size = H.size(0)
            H, A1, A2, V = H.to(device), A1.to(device), A2.to(device), V.to(device)
            atten1,atten2= model.eval_model_attention((H, A1, A2, V, Atom_count), device)
            atten1 = atten1.detach().cpu().numpy()
            atten2 = atten2.detach().cpu().numpy()
            Final_atten1 += list(atten1)
            Final_atten2 += list(atten2)
    return Final_atten1,Final_atten2


#단일 PDB 파일에 대해 주의 가중치를 계산하고, 시각화할 수 있는 형식으로 저장하는 전체 파이프라인을 수행
def visualize_attention(input_path,params):
    #create saving path
    save_path=os.path.join(os.getcwd(),"Predict_Result")
    mkdir(save_path)
    save_path = os.path.join(save_path, "Visulize_Target")
    mkdir(save_path)
    save_path = os.path.join(save_path, "Fold_"+str(params['fold'])+"_Result")
    mkdir(save_path)
    input_path=os.path.abspath(input_path)
    split_name=os.path.split(input_path)[1]
    original_pdb_name=split_name
    if ".pdb" in split_name:
        split_name=split_name[:-4]
    save_path=os.path.join(save_path,split_name)
    mkdir(save_path)

    #load model
    fold_choice = params['fold']
    model_path = os.path.join(os.getcwd(), "best_model")
    model_path = os.path.join(model_path, "fold" + str(fold_choice))
    model_path = os.path.join(model_path, "checkpoint.pth.tar")
    model, device = init_model(model_path, params)

    structure_path = os.path.join(save_path, "Input.pdb")
    shutil.copy(input_path, structure_path)
    input_file = Prepare_Input(structure_path)
    list_npz = [input_file]
    dataset = Single_Dataset(list_npz)
    dataloader = DataLoader(dataset, 1, shuffle=False,
                            num_workers=params['num_workers'],
                            drop_last=False, collate_fn=collate_fn)

    Final_atten1,Final_atten2 = Get_Attention(dataloader, device, model)
    tmp_save_path1 = os.path.join(save_path, "attention1.npy")
    tmp_save_path2 = os.path.join(save_path, "attention2.npy")
    np.save(tmp_save_path1, Final_atten1)
    np.save(tmp_save_path2, Final_atten2)
    #receptor_path=os.path.join(save_path,"Input.rinterface")
    #ligand_path=os.path.join(save_path,"Input.linterface")
    atom_count = 0
    with open(structure_path, "r") as file:
        line = file.readline()
        while line:
            if len(line) > 0 and line[:4] == "ATOM":
                atom_count += 1
            line = file.readline()
    """""
    lcount = 0
    with open(ligand_path, "r") as file:
        line = file.readline()
        while line:
            if len(line) > 0 and line[:4] == "ATOM":
                lcount += 1
            line = file.readline()
    """
    attention1=Final_atten1
    attention2=Final_atten2

    #all_atom = rcount + lcount
    
    if len(attention1) == 1:
        attention1 = attention1[0]
        attention2 = attention2[0]
    print("number of atoms total %d" % atom_count)
    print("attention shape", attention1.shape)
    assert atom_count == len(attention1) and atom_count == len(attention2)
    attention1 = np.sum(attention1, axis=1)
    attention2 = np.sum(attention2, axis=1)

    """"
    new_receptor_path1 = os.path.join(save_path, "attention1_receptor.pdb")
    new_ligand_path1 = os.path.join(save_path, "attention1_ligand.pdb")
    new_receptor_path2 = os.path.join(save_path, "attention2_receptor.pdb")
    new_ligand_path2 = os.path.join(save_path, "attention2_ligand.pdb")
    Write_Attention(receptor_path, new_receptor_path1, new_receptor_path2, attention1[:rcount], attention2[:rcount])
    Write_Attention(ligand_path, new_ligand_path1, new_ligand_path2, attention1[rcount:], attention2[rcount:])
    """
    new_attention_path1 = os.path.join(save_path, "attention1.pdb")
    new_attention_path2 = os.path.join(save_path, "attention2.pdb")
    Write_Attention(structure_path, new_attention_path1, new_attention_path2, attention1, attention2)


#원래의 PDB 파일에 각 원자의 주의 가중치를 포함하여 새로운 PDB 파일을 생성
def Write_Attention(read_path,w_path1,w_path2,attention1,attention2):
    count_atom=0
    with open(w_path1,'w') as wfile1:
        with open(w_path2,'w') as wfile2:
            with open(read_path,'r') as rfile:
                line=rfile.readline()
                while line:
                    if len(line) > 0 and line[:4] == "ATOM":
                        tmp_atten1=attention1[count_atom]
                        tmp_atten2=attention2[count_atom]
                        wline1=line[:60]+"%6.2f\n" %tmp_atten1
                        wline2=line[:60]+"%6.2f\n" %tmp_atten2
                        wfile1.write(wline1)
                        wfile2.write(wline2)
                        count_atom+=1
                    line=rfile.readline()