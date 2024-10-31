#PDB파일 형식의 입력을 처리하여 단백질 사이의 인터페이스를 추출
# PDB 파일에서 원자 정보를 읽고, 수용체와 리간드의 인터페이스를 추출하여 별도의 파일로 저장
import os
from ops.Timer_Control import set_timeout,after_timeout
#RESIDUE_Forbidden_SET는 특정 잔기(예: FAD)를 제외하기 위해 사용,특정 잔기를 출력하지 않도록 하기 위한 설정
RESIDUE_Forbidden_SET={"FAD"}

#Extract_Interface함수의 출력:수용체와 리간드의 인터페이스 파일 경로
def Extract_Interface(pdb_path):
    """
    specially for 2 docking models
    :param pdb_path:docking model path
    :rcount: receptor atom numbers
    :return:
    extract a receptor and ligand, meanwhile, write two files of the receptor interface part, ligand interface part
    """
    #각 원자 정보 저장하는 배열
    receptor_list=[]
    ligand_list=[]
    #각 잔기의 원자 좌표 정보를 저장하는 배열
    rlist=[]
    llist=[]
    #원자 개수를 계산
    count_r=0
    count_l=0
    

    with open(pdb_path,'r') as file:
        line = file.readline()               # call readline()
        while line[0:4]!='ATOM':            #PDB 파일에서 원자 좌표 정보를 나타내는 ATOM 섹션을 찾음
            line=file.readline()
        atomid = 0
        count = 1
        goon = False
        chain_id = line[21]
        residue_type = line[17:20]
        pre_residue_type = residue_type
        tmp_list = []
        pre_residue_id = 0
        pre_chain_id = line[21]
        first_change=True
        b=0
        while line:
            dat_in = line[0:80].split()
            if len(dat_in) == 0:
                line = file.readline()
                continue
            if (dat_in[0] == 'TER'):            #PDB 파일에서 TER 레코드는 체인의 끝을 나타냄
                b=b+1
            if (dat_in[0] == 'ATOM'):           #원자 정보는 체인 ID, 잔기 ID, 좌표(x, y, z), 잔기 유형, 그리고 원자 유형 등을 포함하여 추출
                chain_id = line[21]
                residue_id = int(line[23:26])

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                residue_type = line[17:20]
                # First try CA distance of contact map
                atom_type = line[13:16].strip()
                if b == 0:                                            #수용체로 분류  
                    rlist.append(tmp_list)                            #수용체의 잔기원자좌표
                    tmp_list = []
                    tmp_list.append([x, y, z, atom_type, count_l])
                    count_l += 1
                    receptor_list.append(line)                        #수용체원자데이터를 가짐
                else:                                                 #리간드로 분류
                    llist.append(tmp_list)                            #리간드의 잔기원자좌표
                    tmp_list = []
                    tmp_list.append([x, y, z, atom_type, count_l])
                    count_l += 1
                    ligand_list.append(line)                          #리간드원자데이터를 가짐

                atomid = int(dat_in[1])
                chain_id = line[21]
                count = count + 1
                pre_residue_type = residue_type
                pre_residue_id = residue_id
                pre_chain_id = chain_id
            line = file.readline()
    print("Extracting %d/%d atoms for receptor, %d/%d atoms for ligand"%(len(receptor_list),count_r,len(ligand_list),count_l))
    final_receptor, final_ligand=Form_interface(rlist,llist,receptor_list,ligand_list)
    #write that into our path
    rpath=Write_Interface(final_receptor,pdb_path,".rinterface")
    lpath=Write_Interface(final_ligand, pdb_path, ".linterface")
    print(rpath,lpath)
    return rpath,lpath


@set_timeout(100000, after_timeout)

#입력: 수용체와 리간드의 원자 리스트와 잔기 리스트
#출력: 인터페이스에 포함된 수용체와 리간드의 최종 리스트
def Form_interface(rlist,llist,receptor_list,ligand_list,cut_off=10):
    cut_off=cut_off**2
    #인터페이스에 존재하는 수용체와 리간드 잔기의인덱스를 저장
    r_index=set()
    l_index=set()

    #두 잔기리스트의 각 원자쌍 사이의 유클리드 거리를 제곱하여 cut_off 이하면 r_index, l_index에 추가
    for rindex,item1 in enumerate(rlist):
        for lindex,item2 in enumerate(llist):
            min_distance=1000000
            residue1_len=len(item1)
            residue2_len=len(item2)
            for m in range(residue1_len):
                atom1=item1[m]
                for n in range(residue2_len):
                    atom2=item2[n]
                    distance=0
                    for k in range(3):
                        distance+=(atom1[k]-atom2[k])**2
                    #distance=np.linalg.norm(atom1[:3]-atom2[:3])
                    if distance<=min_distance:
                        min_distance=distance
            if min_distance<=cut_off:
                if rindex not in r_index:
                    r_index.add(rindex)
                if lindex not in l_index:
                    l_index.add(lindex)
    r_index=list(r_index)
    l_index=list(l_index)
    newrlist=[]
    for k in range(len(r_index)):
        newrlist.append(rlist[r_index[k]])
    newllist=[]
    for k in range(len(l_index)):
        newllist.append(llist[l_index[k]])
    print("After filtering the interface region, %d/%d residue in receptor, %d/%d residue in ligand" % (len(newrlist),len(rlist), len(newllist),len(llist)))
    #get the line to write new interface file
    
    
    #필터링된 잔기의 각 원자 정보를 원본 리스트에서 가져와 final_receptor, final_ligand에 저장
    final_receptor=[]
    final_ligand=[]
    for residue in newrlist:
        for tmp_atom in residue:
            our_index=tmp_atom[4]
            final_receptor.append(receptor_list[our_index])
    try:
        for residue in newllist:
            for tmp_atom in residue:
                our_index=tmp_atom[4]
                final_ligand.append(ligand_list[our_index])
    except:
        b=0
        for residue in newllist:
            for tmp_atom in residue:
                our_index=b
                final_ligand.append(ligand_list[our_index])
                b=b+1
    print("After filtering the interface region, %d receptor, %d ligand"%(len(final_receptor),len(final_ligand)))
    print(final_receptor,final_ligand)
    return final_receptor,final_ligand

#입력: 인터페이스 리스트와 PDB 파일 경로, 확장자
#출력: 새로 작성된 파일 경로, RESIDUE_Forbidden_SET에 포함된 잔기는 제외시킴
def Write_Interface(line_list,pdb_path,ext_file):
    new_path=pdb_path[:-4]+ext_file
    with open(new_path,'w') as file:
        for line in line_list:
            #check residue in the common residue or not. If not, no write for this residue
            residue_type = line[17:20]
            if residue_type in RESIDUE_Forbidden_SET:
                continue
            file.write(line)
    return new_path
