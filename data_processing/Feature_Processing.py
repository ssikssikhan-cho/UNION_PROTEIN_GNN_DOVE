import numpy as np
# x : 원소기호 
#특정 값 x가 주어진 allowable_set에 속해 있는지 확인하고, 해당 값을 인코딩하는 함수
#x가 allowable_set에 있는 경우, 해당 위치에 True, 나머지 위치에 False
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))


#x가 allowable_set에 없을 경우 allowable_set의 마지막 값으로 매핑하여 인코딩
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


#개별 원자의 여러 화학적 특성을 벡터로 변환
# m : 분자 객체 (RDKit의 Mol 객체)
#atom.GetSymbol(): 원소 기호 10개의 가능한 기호로 인코딩
#atom.GetDegree(): 해당 원자에 연결된 결합의 수, [0, 1, 2, 3, 4, 5]로 인코딩
#atom.GetTotalNumHs(): 총 수소 원자 수, [0, 1, 2, 3, 4]로 인코딩
#atom.GetImplicitValence(): 암묵적 원자가 수, [0, 1, 2, 3, 4, 5]로 인코딩
#atom.GetIsAromatic(): 원자가 방향족인지 여부, 단일 True/False 값으로 인코딩
def atom_feature(m, atom_i):
    atom = m.GetAtomWithIdx(atom_i)
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])    # (10, 6, 5, 6, 1) --> total 28


#전체 분자에 있는 모든 원자의 특성을 벡터로 변환하여 배열로 반환
# m : 분자 객체 (RDKit의 Mol 객체)
# H : n x 56 크기의 numpy 배열(n : 분자수)
def get_atom_feature(m, is_ligand=True):
    n = m.GetNumAtoms()
    H = []
    for i in range(n):
        H.append(atom_feature(m, i))
    H = np.array(H)
    if is_ligand:         #is_ligand=True 면 H의 오른쪽에 동일한 크기의 0행렬(0으로 채워진 배열)을 추가하여 n x 56 크기로 확장
        H = np.concatenate([H, np.zeros((n,28))], 1)
    else:
        H = np.concatenate([np.zeros((n,28)), H], 1)
    return H