from dataset import *
import numpy as np



K = 100
KK = 200 # the effect from the source to the recipient

def FindingTopK():
    with open('/data/home/mengtong_zhang/impoverished_students/anonymous_data/indexMatch' + year + '.json', 'r',
              encoding='gbk') as f:
        data = json.load(f)
        graph_dict_S = data["source_index"]
        N = len(graph_dict_S)
        dict_R = data['recipient_index']
        graph_dict_R = {v: k for k, v in dict_R.items()}
        M = len(graph_dict_R)

    file_path = '/data/home/mengtong_zhang/impoverished_students/anonymous_data/' + year + 'AttCoeff.npz'
    data = np.load(file_path)
    Coeff12 = data['Coeff12'] * 1e10
    Coeff3 = data['Coeff3'] * 1e10
    Coeff4 = data['Coeff4'] * 1e10

    interAttS = [np.argwhere(row == np.max(row)).flatten().tolist() for row in Coeff12]
    interAttR = [np.argwhere(row == np.max(row)).flatten().tolist() for row in Coeff12.transpose()]

    cityAtt = [np.argwhere(row == np.max(row)).flatten().tolist() for row in Coeff3]

    provinceAtt = [np.argwhere(row == np.max(row)).flatten().tolist() for row in Coeff4]
    '''interAttS = np.argsort(-Coeff12, axis=1)[:, :M]
    interAttR = np.argsort(-Coeff12.transpose(), axis=1)[:, :KK]  # (M,KK)
    cityAtt = np.argsort(-Coeff3, axis=1)[:, :K] #(N,K)
    provinceAtt = np.argsort(-Coeff4, axis=1)[:, :K]  # (N,K)'''

    temp = [len(f) for f in interAttR]#large -> small


    InterAttS = {}
    for i in range(N):
        InterAttS[i] = []
        for j in interAttS[i]:
            index = j
            InterAttS[i].append(graph_dict_R[index])

    InterAttR = {}
    for i in range(M):
        InterAttR[i] = []
        for j in interAttR[i]:
            index = str(j)
            InterAttR[i].append(graph_dict_S[index])

    CityAtt = {}
    for i in range(N):
        CityAtt[i] = []
        for j in cityAtt[i]:
            index = str(j)
            CityAtt[i].append(graph_dict_S[index])

    ProvinceAtt = {}
    for i in range(N):
        ProvinceAtt[i] = []
        for j in provinceAtt[i]:
            index = str(j)
            ProvinceAtt[i].append(graph_dict_S[index])

    #print(InterAttS, InterAttR, CityAtt, ProvinceAtt)

    data = {
        'InterAttS': InterAttS,
        'InterAttR': InterAttR,
        'CityAtt': CityAtt,
        'ProvinceAtt': ProvinceAtt
    }
    '''print(InterAttS[0])
    print(CityAtt[0])
    print(ProvinceAtt[0])'''

    inter = {}
    for k,v in InterAttR.items():
        inter[graph_dict_R[k]] = v
    print(inter)
    #with open('Explainer'+year+'.json', 'w') as file:
    #    json.dump(data, file)


FindingTopK()