import os
from data_processing.Extract_Interface import Extract_Interface #수용체와 리간드의 인터페이스를 추출하는 함수
from rdkit.Chem.rdmolfiles import MolFromPDBFile                #PDB 파일을 읽어 Mol 객체로 변환
from data_processing.Feature_Processing import get_atom_feature #원자의 특성을 인코딩하여 반환하는 함수
import numpy as np 
from rdkit.Chem.rdmolops import GetAdjacencyMatrix              #분자의 인접 행렬을 생성
from scipy.spatial import distance_matrix

#PDB 구조 파일을 처리하여 딥러닝 모델의 입력으로 사용할 수 있는 형태로 데이터를 준비하는 기능
#입력:PDB 구조 파일의 경로 (structure_path)
#출력:입력 데이터를 .npz 형식으로 저장한 파일의 경로
def Prepare_Input(structure_path):
    # extract the interface region
    root_path=os.path.split(structure_path)[0]
    interface_path = Extract_Interface(structure_path)
    

    #RDKit의 MolFromPDBFile 함수로 수용체와 리간드 PDB 파일을 Mol 객체로 변환
    #sanitize=False 옵션은 분자의 구조를 엄격히 검사하지 않고 가져오는 설정
    mol = MolFromPDBFile(interface_path, sanitize=False)
   

    #GetNumAtoms():분자 내 원자의 개수를 반환
    #get_atom_feature:각 분자의 모든 원자에 대해 길이 56의 특성 벡터를 생성하여 반환
    atom_count = mol.GetNumAtoms()
    atom_feature = get_atom_feature(mol)

    # get receptor adj matrix
    # GetConformers():분자의 좌표를 가져오고, 이를 np.array로 변환하여 d1에 저장
    # GetAdjacencyMatrix():원자 간의 결합을 나타내는 인접 행렬을 반환, 대각 행렬 np.eye()를 더하여 자기 자신과의 연결을 나타냄
    c = mol.GetConformers()[0]
    d = np.array(c.GetPositions())
    adj_matrix = GetAdjacencyMatrix(mol) + np.eye(atom_count)

    
    #distance_matrix(d1, d2):수용체와 리간드 사이의 유클리드 거리 행렬을 계산
    dm = distance_matrix(d, d)
    H = atom_feature
    agg_adj =adj_matrix

    valid = np.ones((atom_count,))


#agg_adj2 = np.copy(agg_adj1)
#agg_adj2[:receptor_count, receptor_count:] = np.copy(dm)
#agg_adj2[receptor_count:, :receptor_count] = np.copy(np.transpose(dm))  # with interaction array
    # node indice for aggregation
    # valid 벡터는 수용체 원자=1, 리간드 원자=0
#valid = np.zeros((receptor_count + ligand_count,))
#valid[:receptor_count] = 1


    input_file=os.path.join(root_path,"Input.npz")
    # sample = {
    #     'H': H.tolist(),
    #     'A1': agg_adj1.tolist(),
    #     'A2': agg_adj2.tolist(),
    #     'V': valid,
    #     'key': structure_path,
    # }
    np.savez(input_file,  H=H, A1=agg_adj, A2=dm, V=valid)
    return input_file
