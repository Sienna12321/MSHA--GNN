# coding=gbk
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import csv
import time

year = '2018'

def round_and_scale(tensor):
    scaled_tensor = tensor * 0.1
    rounded_tensor = scaled_tensor.round().long()
    return rounded_tensor

def calculate_indices(tensor):
    rows, cols = tensor.nonzero(as_tuple=True)
    values = tensor[rows, cols]
    source_index = []
    recipient_index = []
    for i in range(len(rows)):
        source_index += [rows[i].item()] * int(values[i].item())
        recipient_index += [cols[i].item()] * int(values[i].item())
    return source_index, recipient_index

class HigherDataset_temp(Dataset):
    def __init__(self):
        self.source = []
        self.county = []
        self.city = []
        self.province = []
        self.school = []
        self.school_province = []
        self.family = []#family poverty status
        self.GDP = {}
        self.majors = []
        self.gender = []
        self.index_match = {}#match the source name, to (county, city, province, GENDER, major, family_poverty_status)
        #match the county with the GDP
        county_GDP = {}
        with open('/data/home/mengtong_zhang/impoverished_students/HGANE/15-18_countyGDP.csv','r',encoding='gbk') as f:
            header = next(f)
            file = csv.reader(f)
            for row in file:
                county_GDP[row[0][:2]] = row[4]#year 2015
        #load data

        count = 0
        with open('/data/home/mengtong_zhang/impoverished_students/higher_data_year/2018.csv','r', encoding='gb18030') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                count+=1
                if(count==3000):
                    break
                if(row[7] == None):
                    continue
                if(row[14]==None):
                    continue
                major_ = majorClassify(row[16])
                if(major_==None):
                    continue
                self.majors.append(major_)

                if (row[3] == '2'):
                    GENDER = 'Ů'
                else:
                    GENDER = '��'
                self.gender.append(GENDER)
                self.city.append(row[6])
                self.province.append(row[5])
                self.county.append(row[7])
                family_poverty_status = row[11]
                S = row[5]+row[6]+row[7]+GENDER+major_+family_poverty_status
                self.source.append(S)
                #match the source name, to (county, city, province, GENDER, major, family_poverty_status)
                self.index_match[S] = [row[7], row[6], row[5], GENDER, major_, family_poverty_status]
                #self.school.append(row[15])
                self.school_province.append(row[14])
                try:
                    self.GDP[S] = int(county_GDP[row[7][:2]])#GDP{source:gdp}
                except:
                    self.GDP[S] = 1000000
        print('finish loadin')
        print(time.ctime())
        #build node embedding
        self.source_index = self.node_index(self.source)#l2
        #self.source_index: {source:index}, self.source_embedding: {source: embedding}
        self.recipient_index = self.node_index(self.school_province)#recipient_embedding:(m,dim)
        self.GDP_embedding = self.gdp_embedding()#(n,) {source_index: gdp_embedding} source index��С�����˳������
        self.city_adj, self.province_adj = self.intra_adjacent()
        self.school_adjacent = self.inter_adjacent()#(n,m)
        print('finish load index dict')
        print(time.ctime())

    def __getitem__(self, index):
        source_index = self.source_index[self.source[index]]
        recipient_index = self.recipient_index[self.school_province[index]]
        return source_index, recipient_index

    def get_adjacent(self):
        return self.school_adjacent, self.city_adj, self.province_adj

    def get_gdp(self):
        return self.GDP_embedding

    def get_count(self):
        #return Scount, Rcount
        Scount, Rcount = self.school_adjacent.size()
        return Scount, Rcount

    def __len__(self):
        return len(self.source)

    def process(self):

        S_Index = {}
        for source, i in self.source_index.items():
            for s, fix in self.index_match.items():
                if(source == s):
                    S_Index[i] = fix
        index = {'source_index':S_Index,
                 'recipient_index':self.recipient_index}

        with open('/data/home/mengtong_zhang/impoverished_students/BaseLine/GCN/indexMatch2016.json', 'w',encoding='utf-8') as file:
            json.dump(index, file)
        print('finish writing')
        print(time.ctime())

    def intra_adjacent(self):
        citys = {}#{city name:[i,j,k,...(source index belong to the key city)],...}
        provinces = {}
        for city in set(self.city):
            citys[city] = []
        for province in set(self.province):
            provinces[province] = []
        for i in range(len(self.source)):
            citys[self.city[i]].append(self.source_index[self.source[i]])
            provinces[self.province[i]].append(self.source_index[self.source[i]])

        index_num = len(self.source_index)
        city_adjacent = [[0 for _ in range(index_num)] for __ in range(index_num)]
        province_adjacent = [[0 for _ in range(index_num)] for __ in range(index_num)]
        for city_name, county_indexs in citys.items():
            for i in county_indexs:
                for j in county_indexs:
                    city_adjacent[i][j] = 1

        for province_name, county_indexs in provinces.items():
            for i in county_indexs:
                for j in county_indexs:
                    province_adjacent[i][j] = 1
        return torch.tensor(city_adjacent,dtype=torch.float32), torch.tensor(province_adjacent,dtype=torch.float32)

    def inter_adjacent(self):
        inter = {}
        for province in set(self.school_province):
            inter[province] = []#contain the source county index

        for i in range(len(self.source)):
            inter[self.school_province[i]].append(self.source_index[self.source[i]])

        n = len(self.source_index)
        m = len(self.recipient_index)
        adjacent = [[0 for _ in range(n)] for __ in range(m)]#(m,n)
        for school_province, county_indexs in inter.items():
            school_index = self.recipient_index[school_province]
            for k in county_indexs:
                adjacent[school_index][k] += 1
        return torch.tensor(adjacent,dtype=torch.float32).t()#(n,m)

    def node_index(self,nodes):
        '''
        build random node embedding
        :param nodes: list containing locations {location name(source): index}, dim: embedding dim
        :return: node_index:{location name(source): index}, nodes_embedding
        '''
        only_nodes = set(nodes)
        index = 0
        node_index = {}
        for node in only_nodes:
            node_index[node] = index#from 0 to n-1
            index += 1
        #nodes_embedding = torch.rand([len(only_nodes), self.dim])
        return node_index#, nodes_embedding

    def gdp_embedding(self):
        '''
        integrate GDP, family_poverty_status, random node embedding to formal node embedding
        :param random_embedding:
        :return: GDP embedding matched with node index {source: embedding}
        '''
        max_GDP = max(self.GDP.values())
        min_GDP = min(self.GDP.values())
        #normalize GDP
        GDP_embedding = {}
        for source,gdp in self.GDP.items():
            g = (gdp-min_GDP) / (max_GDP - min_GDP)
            #gdp = torch.full((1, self.dim), self.GDP[source])
            GDP_embedding[self.source_index[source]] = g
        sorted_GDP = {}#����source index��С�����˳������
        for key in sorted(GDP_embedding.keys()):
            sorted_GDP[key] = GDP_embedding[key]
        return sorted_GDP


class HigherDataset(Dataset):
    def __init__(self):
        self.source = []
        self.recipient = []
        self.city = []
        self.province = []
        self.GDP = []#ÿ��ѧ�������ص�GDP����һ����
        self.count = 0
        with open('/data/home/mengtong_zhang/impoverished_students/anonymous_data/GDP'+year+'.json','r',encoding='gbk') as f:
            data = json.load(f)
            self.GDP = data["GDP_embedding"]#{"0": 0.030797779997757676, "1": 0.03228721668651744, "2":...}

        with open('/data/home/mengtong_zhang/impoverished_students/anonymous_data/indexMatch'+year+'.json','r',encoding='gbk') as f:
            data = json.load(f)
            self.graph_dict = data["source_index"]
            self.N = len(self.graph_dict)
            self.graph_dict_R = data['recipient_index']
            self.M = len(self.graph_dict_R)

        with open('/data/home/mengtong_zhang/impoverished_students/anonymous_data/Flow'+year+'.csv','r', encoding='gb18030') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                self.count += 1
                self.source.append(int(row[0]))
                self.recipient.append(int(row[1]))
                self.city.append(int(row[2]))
                self.province.append(int(row[3]))

        self.city_adj, self.province_adj = self.intra_adjacent()
        self.school_adjacent = self.inter_adjacent()#(n,m)


    def __getitem__(self, index):
        source_index = self.source[index]
        recipient_index = self.recipient[index]
        return source_index, recipient_index

    def get_adjacent(self):
        return self.school_adjacent, self.city_adj, self.province_adj

    def get_gdp(self):
        return self.GDP

    def get_count(self):
        #return Scount, Rcount
        Scount, Rcount = self.school_adjacent.size()
        return Scount, Rcount

    def __len__(self):
        return len(self.source)

    def intra_adjacent(self):
        #(county, city, province, GENDER, major, family_poverty_status)
        num_nodes = len(self.graph_dict) # N
        city_adjacent = torch.zeros(num_nodes, num_nodes)
        province_adjacent = torch.zeros(num_nodes, num_nodes)

        # �����ֵ䣬�����ڽڵ��λ����Ϊ1
        for i, (node, values) in enumerate(self.graph_dict.items()):
            for j, (neighbor, neighbor_values) in enumerate(self.graph_dict.items()):
                if values[1] == neighbor_values[1]:
                    city_adjacent[i, j] = 1

        for i, (node, values) in enumerate(self.graph_dict.items()):
            for j, (neighbor, neighbor_values) in enumerate(self.graph_dict.items()):
                if values[2] == neighbor_values[2]:
                    province_adjacent[i, j] = 1

        return torch.tensor(city_adjacent,dtype=torch.float32), torch.tensor(province_adjacent,dtype=torch.float32)

    def inter_adjacent(self):
        inter_adjacent = torch.zeros(self.N, self.M)
        inter = {}
        for i in range(self.N):
            inter[i] = {}
            for j in range(self.M):
                inter[i][j] = 0
        for i in range(self.count):
            inter[self.source[i]][self.recipient[i]] += 1
            inter_adjacent[self.source[i]][self.recipient[i]] += 1

        data = {
            'inter': inter
        }
        with open('/data/home/mengtong_zhang/impoverished_students/anonymous_data/inter' + year + '.json', 'w') as file:
            json.dump(data, file)

        return torch.tensor(inter_adjacent,dtype=torch.float32)#(n,m)

    def node_index(self,nodes):
        '''
        build random node embedding
        :param nodes: list containing locations {location name(source): index}, dim: embedding dim
        :return: node_index:{location name(source): index}, nodes_embedding
        '''
        only_nodes = set(nodes)
        index = 0
        node_index = {}
        for node in only_nodes:
            node_index[node] = index#from 0 to n-1
            index += 1
        #nodes_embedding = torch.rand([len(only_nodes), self.dim])
        return node_index#, nodes_embedding

    def gdp_embedding(self):
        '''
        integrate GDP, family_poverty_status, random node embedding to formal node embedding
        :param random_embedding:
        :return: GDP embedding matched with node index {source: embedding}
        '''
        max_GDP = max(self.GDP.values())
        min_GDP = min(self.GDP.values())
        #normalize GDP
        GDP_embedding = {}
        for source,gdp in self.GDP.items():
            g = (gdp-min_GDP) / (max_GDP - min_GDP)
            #gdp = torch.full((1, self.dim), self.GDP[source])
            GDP_embedding[self.source_index[source]] = g
        sorted_GDP = {}#����source index��С�����˳������
        for key in sorted(GDP_embedding.keys()):
            sorted_GDP[key] = GDP_embedding[key]
        return sorted_GDP

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    #preds = output.max(1)[1].type_as(labels)
    #correct = preds.eq(labels).double()
    correct = output.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

major = {}  # {third classification: first classification}
with open(r'/data/home/mengtong_zhang/impoverished_students/HGANE/major.csv', 'r', encoding='gb18030', errors='ignore') as f:
    f_csv = csv.reader(f)
    header = next(f_csv)
    for row in f_csv:
        major[row[3][:2]] = row[4].split(',')[0]
more_major = {'�ɻ������豸ά��':'��ѧ', '��ҵ':'ũѧ', '�ܵ����̼���':'��ѧ', '�����ѧ':'��ѧ', '���Ƚ���ѧ':'����ѧ', '����װ�����':'����ѧ', '���⺺���ѧ':'����ѧ', '�����������ż���':'��ѧ', 'ú��������':'��ѧ', '�ط�����ѧ':'����ѧ', '���ܷ����豸������ά��':'��ѧ', '���������뻷��':'��ѧ', '�ִ���֯����':'��ѧ', '���������':'��ѧ', 'ũ��Ʒ��ȫ�뻷��':'ũѧ', '���ܿ�ѧ�빤��':'��ѧ', '��ҩ��ҩ����':'ҽѧ', 'ý�龭Ӫ�����':'����ѧ', '�𽺹��̼���':'��ѧ', '�����봫ý':'��ѧ', 'Ͷ���ʹ���':'����ѧ', 'ý��������Ǩ':'��ѧ', '��ͯ��ѧ':'��ѧ', '��ҵ':'ũѧ', '���򻷾�ѧ':'��ѧ', '��̨�������������':'����ѧ', 'ģ��������':'����ѧ', '�������̼���':'��ѧ', '��������ѧ�����ѧ':'ũѧ', '�����豸Ӧ����ά��':'��ѧ', '���蹤�̼���':'��ѧ', '����ղ��뿱̽':'��ѧ', '��������Դ��ѧ�빤��':'��ѧ', '��·������۹���':'��ѧ', '�о�������ѧ':'����ѧ', 'ˮ��������뱣��':'��ѧ', '������յ�����':'��ѧ', '�Ƚ��߷��Ӳ���':'��ѧ', '��������':'����ѧ', '����ý�����':'����ѧ', '���۹滮���':'����ѧ', '�˼�������Ϲ���':'��ѧ', '���ڡ��Զ������뵯ҩ����':'��ѧ', '���ﻯѧӦ�ü���':'��ѧ', '�Ƿ�������ѧ':'��ѧ', '�����������':'����ѧ', '���������ʼ켼��':'��ѧ', '������':'����ѧ', '����':'��ѧ', '��������빤��':'����ѧ', '�糧�豸������ά��':'��ѧ', '�޻��ӹ��뾭Ӫ����':'����ѧ', '��ѧ':'��ѧ', '��������ѧ':'��ѧ', '�ƶ�Ӧ�ÿ���':'��ѧ', 'Ƥ�������빤��':'����ѧ', '��ͼѧ�������Ϣϵͳ':'��ѧ', 'ҩƷ������':'ҽѧ', '�꾮����':'��ѧ', '���������������':'����ѧ', 'Ƥ�����':'����ѧ', '��С��ҵ��ҵ�뾭Ӫ':'����ѧ',
             '�����Ӧ������':'����ѧ', 'Ԫ��ƻ�ʵ���':'����ѧ', '�ṹ����':'��ѧ', '��ͧ���������':'��ѧ', '�Ӳ�ѧ':'ũѧ', 'Ȩ����Ϣ������':'����ѧ', 'ũ��Ʒ������ʳ�ﰲȫ':'ũѧ', '����������ά��':'��ѧ', '�ɻ������豸���Ǳ�':'��ѧ', '����ѧ':'ũѧ', '��ѧ�뷭��':'��ѧ', '���÷�֯Ʒ���':'����ѧ', '�������ѧ':'��ѧ', '�ڿ�ѧ':'ҽѧ', '��ý�߻������':'����ѧ', '��ͼ��ͼ�����ִ�������':'��ѧ', '����ʯ������ӹ�':'��ѧ', '��������ѧ':'��ѧ', '����':'����ѧ', '���ͽ������ϼ���':'��ѧ', '��ɽ����':'��ѧ', '�Ṥ��':'��ѧ', '��֯��������֯��װ':'��ѧ', 'ũ��Ʒ�������':'ũѧ', '�Ƽ��㼼����Ӧ��':'��ѧ', '��չ����ѧ':'����ѧ', '����ϵͳ�����Զ���':'��ѧ', 'ˮ��Դ�����뱣��':'��ѧ', '����ӹ�����':'��ѧ', '������':'����ѧ', '���ѧ':'����ѧ', 'ȼ�ն���ѧ':'��ѧ', '�ȴ�԰������ѧ':'ũѧ', '������ѧ':'��ѧ', '̫���ܹ��ȼ�����Ӧ��':'��ѧ', '��⼼����Ӧ��':'��ѧ', '�̷�ѧ':'��ѧ', '�������˼�����о�':'��ѧ', '���ߵ�����':'��ѧ', '������·���˳���':'����ѧ', '�����깤�������':'����ѧ', '�ǽ���������ϼ���':'��ѧ', '��ҵ���ù���':'����ѧ', 'ҩ������ѧ':'ҽѧ', '����ѧԺʵ�����':'����ѧ', '�繤�������¼���':'��ѧ', '��������':'��ѧ', '������缼����Ӧ��':'��ѧ', '�������ྻ��Դ':'��ѧ', '�ɻ��ṹ����':'��ѧ', '��������������ѧ':'ҽѧ', 'չʾ�������':'����ѧ', 'ˮ��վ�����豸':'��ѧ', '�˼�����Ӧ��':'��ѧ', '��ý��ѧ':'��ѧ', '��ý����ѧ':'����ѧ', '��������':'��ѧ', '����ѧ':'ũѧ', '�ݵ����������':'ũѧ', 'ˮ���ṹ����':'��ѧ', '�Ƚ�����':'��ѧ', '�糧��ѧ':'��ѧ', '��ɫұ����':'��ѧ',
             'ϸ������ѧ':'��ѧ', '���Ʒ�ӹ�����':'��ѧ', 'ũ��Ʒ������ȫ':'ũѧ', '������ѧ':'��ѧ', '΢�߶ȿ�ѧ�뼼��':'��ѧ', 'ѧ�ƽ�':'����ѧ', 'ҽҩӪ��':'����ѧ', '¥�����ܻ����̼���':'��ѧ', '���ѧ':'ҽѧ', '���糧������ϵͳ':'��ѧ', '��ֳ':'ũѧ', '�������������':'����ѧ', '����ѧ':'��ѧ', '�������̼���':'��ѧ', '����ұ���豸Ӧ�ü���':'��ѧ', '���͹���':'����ѧ', 'ҩ��ֲ����Դ����':'ũѧ', 'д��ѧ':'��ѧ', '����༭��У��':'��ѧ', '��ͯ��ǻҽѧ':'ҽѧ', '�ȽϽ���ѧ':'����ѧ', '����ҵ���ù���':'����ѧ', 'Ⱦ������':'��ѧ', '��ͨ���밲ȫ':'��ѧ', 'ũ��Ʒ�ӹ������ع���':'��ѧ', '��ҵ����':'��ѧ', '�����Ļ�ѧ':'��ѧ', '����ʯ������ӹ�����':'��ѧ', 'һ����ѧ����ѧ����':'��ѧ', 'ѧ�ƽ�ѧ':'����ѧ', '�ҿ���������':'��ѧ', '��ȼ��������ά��':'��ѧ', '�ִ�ũҵ����':'ũѧ', '������Դ���������':'����ѧ', '��������ѧ':'��ѧ', '�й���ʷ':'��ʷѧ', '�Ƚ�����':'��ѧ', '�Ƚ���ѧ��������ѧ':'��ѧ', '������������':'ũѧ', 'ˮ��Ϣѧ':'��ѧ', '��������':'����ѧ', '΢����ѧ':'��ѧ', '�������':'����ѧ', '�ֲ�ʽ������΢��������':'��ѧ', '��ͨ�ż���':'��ѧ', '����ָ��ѧ':'��ѧ', '����ѧ':'����ѧ', '��ز���ѧ���������':'��ѧ', '�Ṥ�����빤��':'��ѧ', '��������':'��ѧ', '���ʹ������⼼��':'ũѧ', '���ļ͵���ѧ':'��ѧ', '�ۿ�ѧ':'ҽѧ', '����ͨ��':'��ѧ', '��ҵ���ѧ':'����ѧ', '���ϵͳ������ά��':'��ѧ', '�Ƚ��ƶ�ѧ':'��ѧ', '��⼼�����Զ���װ��':'��ѧ', '��ҵ':'ũѧ', '�γ����ѧ��':'����ѧ', '���ݹ��̼���':'��ѧ', 'ҩƷ��������':'ҽѧ', '�����������':'����ѧ', '��ϸ��������ҽѧ':'ҽѧ', '��Դ�任������Ӧ��':'��ѧ', '�ߵ�ѹ���Ե����':'��ѧ',
              '���蹤�̹�����':'����ѧ', '���������븾�ױ���ѧ':'ҽѧ', 'װ���������':'����ѧ', '��ҵ����ѧ':'����ѧ', '��ѧ����':'��ѧ', '������������':'����ѧ', 'ˮ��վ�����豸���Զ���':'��ѧ', '�����ʺ��ѧ':'ҽѧ', '�ؾ�':'��ѧ', '����ִ��':'��ѧ', '����ʩ����':'��ѧ', '��ҩѧ':'ҽѧ', '����չ�����滮':'����ѧ', '������ҩ����':'ҽѧ', 'ũ��Ʒ�ӹ����������':'ũѧ', '��������ʻ���':'����ѧ', '������Ŀ��Ϣ������':'����ѧ', '�׶���չ�뽡������':'����ѧ', '��������':'����ѧ', '�����������⿪�ŷ��������о�':'��ѧ', '����ϵͳ�̵籣�����Զ�������':'��ѧ', 'ʦ����':'����ѧ', '���ܶ����豸��Ӧ��':'��ѧ', '���������������ѧ':'��ѧ', '��ѹ�������·ʩ��������ά��':'��ѧ', '�����ɹ�����':'����ѧ', '���ٶ������ʻ':'��ѧ', 'ת�쾭��ѧ':'����ѧ', 'ҩ��ֲ��ѧ':'ũѧ', '�ź�����Ϣ����':'��ѧ', '���ͷ�֯���缼��':'��ѧ', '�������崫ͳ����':'����ѧ', '�����������֯��̥ѧ':'ҽѧ', '���������Զ�������':'��ѧ', '����ͨ����յ����̼���':'��ѧ', '������ѧ�빤��':'��ѧ', 'ҽҩ��Ϣ����':'����ѧ', '����ά��':'��ѧ', '��ҵ����':'����ѧ', '���估���¹���':'��ѧ', '�������������':'��ѧ', '�����붯������':'��ѧ', 'ҩ��ֲ����Դ':'ũѧ', '��������':'��ѧ',
              '�ർ':'����ѧ', '˰��':'����ѧ', '�����벥��':'����ѧ', '���͹���':'��ѧ', '����ϵͳ�Զ�������':'��ѧ', '����':'ũѧ', '�ǲ������ӹ�������':'ũѧ', '��ҵ������':'��ѧ', '������Ϣ����ѧ':'��ѧ', '�׵��ѧ�뼼��':'��ѧ', '�����������':'��ѧ', '��Ȩ��ѧ':'��ѧ', '��ҵ��չ':'ũѧ', '�����Ŵ�����':'ũѧ', '����ѧ':'��ѧ', '�������':'����ѧ', '���������滮':'��ѧ', 'ֱ������ʻ����':'��ѧ', '�л���ѧ':'��ѧ', '������̬�滮�����':'��ѧ', '�����Ǳ���':'��ѧ', '������̬ѧ':'��ѧ', '�ִ����Ἴ�������':'����ѧ', '�Ŀ��������':'����ѧ', '���������뼼��':'��ѧ', '��ѧ��Ϣ����':'��ѧ', '���������Ļ��Ƚ��о�':'��ѧ', '�����ͯ��֯��˼����ʶ��չ':'����ѧ', '��ѧ��':'��ѧ', '�ǲ�ѧ':'ũѧ', '�˿�ѧ':'����ѧ', '��������ѧ':'��ѧ', '��ҵ����':'��ѧ', '��ͨ����⴫��':'��ѧ', '��ͨ�ż���':'��ѧ', '�߶������˶������':'����ѧ', '���칤��':'��ѧ', '��վ�滮�뿪������':'��ѧ', '�󾮽���':'��ѧ', '�Ž������̼���':'��ѧ', '���켼��':'��ѧ', '�������':'����ѧ', '��ʦ��չ�����':'����ѧ', '��������':'����ѧ', '����繤����Ӧ��':'��ѧ', '�����������̼���':'��ѧ', '���ؼ���':'��ѧ', '�ܷ�ѧ��������ѧ':'��ѧ', 'Ƥ�������Բ�ѧ':'ҽѧ', '����ѧ����Ӽ���':'��ѧ', '��Ȼ���ﻯѧ':'��ѧ', '��������ѧ':'��ѧ', '�����������':'����ѧ', 'ҩ��ѧ':'ҽѧ', '�ƽ���ֽ����':'��ѧ', 'Ʒ�ƴ���Ӫ':'����ѧ', 'Ӫ����ʳƷ����ѧ':'ҽѧ', '��ҵ��Ϣ���������':'����ѧ', 'ˮ��ˮ��Դ����':'����ѧ', '���澫�ι���':'��ѧ', '��ѧ':'ũѧ',
              'ŷ��ѧ':'��ʷѧ', '���ɾ���ѧ':'����ѧ', '����ѧ':'��ʷѧ', '�����������':'����ѧ', 'ͼ��ͼ������':'����ѧ', '���򾭼�ѧ':'����ѧ', '��׼ҽѧ�빫������':'ҽѧ', '�ۺ��Ŀƽ���':'����ѧ', '�Ƚ���ѧ���Ļ�����':'��ѧ', '�����������ѧ�о�':'����ѧ', 'ҩƷ���＼��':'ҽѧ', '��������ѧ':'��ѧ', '�ڵ缼�������':'����ѧ', '������Ʊ�����':'��ѧ', '����ʳƷ�ӹ��뿪��':'ũѧ', 'ҽҩ����ѧ':'ҽѧ', 'ˮ��վ���������':'��ѧ', '��ѹ��ѧ�뼼��':'��ѧ', '��������빤��':'����ѧ', '��չ���������ѧ':'����ѧ', 'Ӫ�����':'����ѧ', '��ҩѧ':'ũѧ', '����ѧ':'ũѧ', 'ú̿��ӹ�������':'��ѧ', '������缼����Ӧ��':'��ѧ', '��ɫ����ұ��':'��ѧ', '���������ƶ�':'��ѧ', '�糧��ѧ�뻷������':'��ѧ', '������Ч����':'����ѧ', '����ѧ':'ũѧ', '�������������':'��ѧ', '����������':'��ѧ', 'ҩ��ֲ����Դ������':'ũѧ', '��ɫ��Դ��ѧ�뼼��':'��ѧ', 'ͨ�ú��պ�����':'��ѧ', '����������':'����ѧ', '����������ά��':'����ѧ', '������ϼӹ���Ӧ�ü���':'��ѧ', '��Ȼҩ�ﻯѧ':'ҽѧ', 'ĸӤҽѧ':'ҽѧ', '�ṹ����ѧ':'��ѧ', '�������������':'����ѧ', '�����ѧ':'����ѧ', '������ѧ':'ҽѧ', '�����о�����ѧ����':'��ѧ', 'ԭ�����������':'��ѧ', '�ߵȽ���ѧ':'����ѧ', '��������޼���':'��ѧ', '��������':'��ѧ', '��ҵó����Ͷ��':'����ѧ', 'Ƥ�ﻯѧ�빤��':'��ѧ', 'ˮ��վ�������':'��ѧ', '����ҽѧ':'ҽѧ', 'Ӫ����ʳƷ��ȫ':'����ѧ', 'ת��ҽѧ':'ҽѧ', '��������빫������':'����ѧ', 'Ϸ������':'����ѧ',
              '�����������߲�����':'ũѧ', '�������＼��':'ũѧ', '���Լ�������������':'��ѧ', '�ۺϻ�е����ú':'��ѧ', '��Ϸ���':'����ѧ', '���ӽ�Ŀ����':'����ѧ', '��ϸ��������ҽѧ':'ҽѧ', '�����������':'����ѧ', '�������繤�̼���':'��ѧ', 'ˮ��ѧ����������ѧ':'��ѧ', '����������':'����ѧ', '��������Ա༭����':'��ѧ', '�ǳ�����������ѧ':'��ѧ', '�ۺ�������':'����ѧ', '����ʷ':'��ѧ', '���Ƚ���':'����ѧ', '��ҵ[����ҵ':'ũѧ', 'չʾ���':'����ѧ', '�����뻪�Ľ���':'����ѧ', '��ҵ��':'��ѧ', '���˽���ѧ':'����ѧ', '����������ṹ����':'��ѧ', '���ٶ�������޼���':'��ѧ', '��������Դ�������Դ':'��ѧ', 'Ӧ������ѧ':'��ѧ', '��ͨ�������Ϣ����':'��ѧ', 'ȫ��ҽѧ':'ҽѧ', '��ɽ����':'��ѧ', 'ˮ·�����뺣�¹���':'����ѧ', '����ָ����ս��':'����ѧ', '��·����':'��ѧ', '����������ԭ�Ӻ�����':'��ѧ', '��ɽ���缼��':'��ѧ', '��ʦ����':'����ѧ', '�ܹ�ʩ������':'��ѧ', 'ý��Ӫ��':'����ѧ', '����ҵ��Ϣ������':'��ѧ', '������Դ����������':'ũѧ', 'ѡú����':'��ѧ', '�鱨ѧ':'����ѧ', '�߶�����������':'����ѧ', '��ȼ��ѭ�������':'��ѧ', '������ѧ�뼼��':'��ѧ', '�ݵ�Ӫ������ѧ':'ũѧ', '��������빤��':'����ѧ', '��ɫ��ѧ�뼼��':'��ѧ', '�ɻ����켼��':'��ѧ', '������ѧ�뼼��':'��ѧ', '�ȹ���⼰���Ƽ���':'��ѧ', '����':'ũѧ', '�л�������������':'��ѧ', '�Ṥ��֯ʳƷ��':'��ѧ', '��װ���������':'����ѧ', '����ѧ':'��ѧ', '����������ѱ��':'����ѧ', '��ɽ�ռ���Ϣ����':'��ѧ', '����ѧ�벡������ѧ':'ҽѧ', '׳ѧ����ѧ':'��ʷѧ', '�Ƚ�ҽѧ':'ҽѧ', '������¼':'��ѧ', '��ɫ��װ���̼����':'����ѧ', '����':'��ѧ', '������':'��ѧ', '��ý�����������':'����ѧ',
              '�ƽ���ֽ����':'��ѧ', '������Ƽ���':'����ѧ', '���ڻ����������Ƽ���':'��ѧ', '�ྻ��Դ��ѧ':'��ѧ', '�����������������':'��ѧ', '���������Ӫ��':'����ѧ', '�䶾���μ���':'ҽѧ', '�ƶ�����':'����ѧ', 'ľ�����ϲ��Ͽ�ѧ�빤��':'��ѧ', 'ͬ�����估Ӧ��':'��ѧ', '����ѧ':'ҽѧ', '��Ѫҽѧ':'ҽѧ', '�����ͯ��֯��˼����ʶ����':'����ѧ', '��Ϸ���������':'����ѧ', '��ˮ��ˮ����':'��ѧ', '�ǳ�����������':'��ѧ', '��������ѧ':'����ѧ', '�߽�����ѧ':'��ѧ', '��Ⱦ�޸�����̬���̼���':'��ѧ', '������ѧ':'��ѧ', '����ҽѧ':'ҽѧ', '������������ƹ���':'��ѧ', '���ٵ���':'����ѧ', 'ú�����������':'��ѧ', '����ѧ':'ҽѧ', 'ҩƷ�����밲ȫ':'ҽѧ', 'ͳһս��ѧ':'��ѧ', '��ɫұ����':'��ѧ', 'ˮ�������������':'��ѧ', '�л��軯ѧ������':'��ѧ', '��繤��':'��ѧ', 'Ь������빤��':'����ѧ', 'ͨ�ú�����ά��':'��ѧ', '������':'����ѧ', '��ҵ��Դ':'ũѧ', '�߶����˶����������':'����ѧ', '���׼���':'��ѧ', 'ˮ��������':'��ѧ', '�˳�ѧ�������':'��ѧ', '���ѧ':'����ѧ','���ﰲȫ����':'ũѧ', '�������������ڽ̷��������о�':'��ѧ', '������Ӫ����':'����ѧ', '�߽�ѧ':'��ʷѧ', '��ҵ�Ļ�������':'��ѧ', '��ý���ѧ':'��ѧ', 'άҩѧ':'��ѧ', '֤ȯ���ڻ�':'����ѧ', '�Ŵ�ѧ':'��ѧ', '������':'��ѧ', '�񺽿��а�ȫ����':'��ѧ', '����������':'��ѧ', '�̻������������':'��ѧ', '����԰��':'ũѧ', '��ɫ��ѧ��������Դ�ۺ�����':'��ѧ', '�����ҵ��������':'����ѧ', '��ɫ��ѧ':'��ѧ', '��������ѧ':'��ѧ', '��ҵװ��ά������':'��ѧ', '��ɫʳƷ���������':'ũѧ', 'ũ���ز�Ʒ�ӹ�':'ũѧ', '��������':'��ѧ', '�����뷢�繤����':'��ѧ', '����ѧ':'ҽѧ', '������ҽ':'ũѧ',
              '�����Ļ�����ѧ':'��ѧ', '������˼�������':'��ѧ', '����ִ����':'��ѧ', '��������뿱��':'��ѧ', '���ߵ������켼��':'��ѧ', '����������ʵ��':'��ѧ', 'ͼ����Ϣ����':'��ѧ', '��������ؼ���':'��ѧ', 'ũ��Ʒ��ȫ':'ũѧ', '��·���������':'����ѧ', '���������뼼��':'����ѧ', '���Ϸ�ѧ':'��ѧ', '�����������ѧ':'��ѧ', '�ػ�ѧ':'��ʷѧ', '�������̾��������':'����ѧ', '��������ѧ':'����ѧ','���ܹ���':'��ѧ', '�糧���ܶ���װ��':'��ѧ', '�ƶ�������Ӧ�ü���':'��ѧ', '�߲�ѧ':'ũѧ', 'Ƥ���������':'����ѧ', '��Ϸ���':'����ѧ', '��ҵ����ѧ':'����ѧ', '���⺺��':'��ѧ', '�ۺ������������':'��ѧ', '����Ʒ���������':'����ѧ', '��ԭ����ѧ':'��ѧ', '��ˮ�����빤��':'��ѧ', '����չ�������':'����ѧ', '��˾����ս�������':'����ѧ', '��ҩ������Ӫ��':'����ѧ', '��ͨ����ѧ':'����ѧ', 'Ӫ������':'����ѧ', 'ҩƷ������⼼��':'��ѧ', '����԰��ѧ':'ũѧ', '����ʳƷ��ȫ����':'����ѧ', '���������豸�������Զ���':'��ѧ', '��糧��������':'��ѧ', '��ҵ����':'ũѧ', '�Ƚ���ѧ����Ļ��о�':'��ѧ', 'Ӫ�������':'��ѧ', '�̼켼��':'����ѧ', '���ȡ���ȼ����ͨ�缰�յ�����':'��ѧ', '�Ṥװ��������':'��ѧ', 'ˮ�ֺ���ˮ��ȫ':'��ѧ', '������Դ����������':'����ѧ', '��Ҷ�����ӹ�����':'ũѧ', '��������':'��ѧ', '�ǳ������������뿱̽':'��ѧ', 'Ƕ��ʽ������Ӧ��':'��ѧ', '����ϵͳ�̵籣�����Զ���':'��ѧ', '��������ѧ':'����ѧ', '��·���̼�⼼��':'��ѧ', '�������':'����ѧ', '���ι���':'����ѧ', 'ˮ��ˮ�繤��':'��ѧ',
            '��ѧ':'��ʷѧ', '�������ѧ':'����ѧ', '���Ʒ�ӹ��밲ȫ':'ũѧ', '���蹤�̹���':'����ѧ', '�ߵȼ���·ά�������':'��ѧ', '����̬����':'��ѧ', 'ˮ��Ϣ����':'��ѧ', '��ɢ����ϵͳӦ����ά������':'��ѧ', '֤ȯͶ�������':'����ѧ', '�糧�ȹ��Զ�������':'��ѧ', 'װ���������':'����ѧ', '�ﷸ����������������':'��ѧ', '���ǹ���':'��ѧ', 'ģʽʶ��������ϵͳ':'��ѧ', '�����Ӫ����':'����ѧ', '��ѧ':'��ѧ', '���Ļ������뷭��':'��ѧ', '������ѧ':'��ѧ', '����ѧ':'��ѧ', '��ѧ':'��ѧ', 'ú������뿱�鼼��':'��ѧ', '��ѧ':'��ѧ', 'ũ��Ӫ�������':'ũѧ', 'ѡ����':'��ѧ', '��Ч����':'��ѧ', '�ɻ�ά��':'��ѧ', '�������ҶӪ��':'ũѧ', '�������ʻ����':'��ѧ', '��������':'��ѧ', '�������Ƽ���':'ҽѧ', '��ʴ��ѧ�����':'��ѧ', 'ר��ʷ':'��ʷѧ', 'Ϸ��':'����ѧ', '����ά�������':'��ѧ', '��̨Ӱ�Ӽ���':'����ѧ', '�����е������':'��ѧ', '��Ԯ����':'��ѧ', '����ʵ����':'��ѧ', 'ú�󿪲ɼ���':'��ѧ', '���Ѿ���ѧ':'����ѧ', '����˶ʿ':'��ѧ', 'ͨ�ú��շ��������������':'��ѧ', '�ָ���̬ѧ':'��ѧ', '�˿�ѧ�뼼��':'��ѧ','��������ѧ':'��ѧ', '������':'��ѧ', '��������':'����ѧ', '��Į��̬ѧ':'��ѧ', '����ұ��':'��ѧ', '��ѧ����':'��ѧ', '����滮':'��ѧ', '������':'��ѧ', '���ֹ��̼���':'��ѧ', '̨���о�':'��ʷѧ', '���ƿ�ѧ�빤��':'��ѧ', '������':'����ѧ', '����ָ�������':'����ѧ', '�ؼ���������ع�����Ϣ����':'��ѧ', '3S����������Ӧ��':'��ѧ', '��ý����ѧ':'����ѧ', '������·��������':'����ѧ', '��Ⱦ��ѧ':'ҽѧ', '�����뾰��԰��':'ũѧ', '���ͼ���':'��ѧ', '����ѧ':'ũѧ', 'ý���Ļ�����':'��ѧ', '������Ϣ����':'��ѧ',
              '����������������':'ҽѧ', '��ҵ���ù���':'����ѧ', 'ú����':'��ѧ', '���������Ҿ����':'����ѧ', '�������':'����ѧ', '�����������������':'����ѧ', '��������������':'ũѧ', '��������ѧ':'ҽѧ', '������������������Ϣ����':'��ѧ', '�Ƚ��������Ʊ�����':'��ѧ', '���ڽ���':'����ѧ', '�ֽṹ���켼��':'��ѧ', '����������Դѧ':'ũѧ', '��·��ϵͳ':'��ѧ', '�������������뼼��':'��ѧ', '�ߵ����Ľ���':'����ѧ', 'Һѹ����������':'��ѧ', '��ֳҽѧ':'ҽѧ', '΢����������ҩѧ':'ҽѧ', '��������Ʊ�����':'��ѧ', '����ѧ':'��ѧ', '��̽����':'��ѧ', '�л�����ѧ':'��ѧ', 'ũ��Ʒ��ͨ�����':'����ѧ', 'ģ�����������':'��ѧ', '�����ͻ����������':'����ѧ', '�ִ���������':'����ѧ', 'ҽҩ������ѧ':'��ѧ', '��·��е��ʩ������':'��ѧ', '��ҵ��Ϣ���������':'����ѧ', '��������ѧ':'����ѧ', '�����羭Ӫ����':'����ѧ', '������ѧ':'��ѧ', '�ɻ���������':'��ѧ', '�񺽰�ȫ��������':'����ѧ', '�������﹤��':'ũѧ', '��������ѧ':'ũѧ', '������������ͳ��':'��ѧ', '��ѧ':'��ѧ', 'Ӫ����л����ѧ':'ҽѧ', 'Ӱ��ҽѧ���ҽѧ':'ҽѧ', 'ҩƷ��Ӫ�����':'����ѧ', '��ԭѧ':'ũѧ', '���ƹ���':'��ѧ', '�ִ���������':'����ѧ', 'ҩ��ѧ':'ҽѧ', 'Ӫ����߻�':'����ѧ', '�������������':'����ѧ', '�˻��뻷������':'��ѧ', '��������ѧ':'��ѧ', '���񹤳̼���':'��ѧ', 'ѭ֤ҽѧ':'ҽѧ', '������������':'��ѧ', '�ȽϷ�ѧ':'��ѧ', 'ҩƷ���������':'����ѧ', '�����⼼��':'��ѧ', '�����������Ҷ�ӹ�':'ũѧ', '��ҵ��������Ϣ����':'��ѧ', '������Ϣ������':'��ѧ', '�ؼ���������ع���':'����ѧ', '����������':'��ѧ', '�����������':'����ѧ', '��·��������':'��ѧ', '�ӿں���ѧ':'��ѧ',
              '��������Դ�뻷��':'��ѧ', '�����������ʻ���':'��ѧ', 'ʱ������ѧ':'��ѧ', '���˹������ù���':'��ѧ', '���������봫ý��':'��ѧ', '�޸���̬ѧ':'��ѧ', '�ƶ�����Ӧ�ü���':'��ѧ', '������е������':'��ѧ', '���ٰ���':'����ѧ', '��ҵ���������':'����ѧ', '��������ѧ':'��ѧ', '��ľ�Ŵ�����':'ũѧ', '��ɽ����':'��ѧ', '������������':'����ѧ', '�к������뻷����ȫ':'��ѧ', '���������ҵ����':'����ѧ', '��ͼ��ͼѧ�������Ϣ����':'��ѧ', '�������':'��ѧ', '�����������������':'��ѧ', '̫���ܹ��ȼ�����Ӧ��':'��ѧ', '�����ֳ���ά��':'��ѧ', '���г���':'����ѧ', '���̷�ѧ':'��ѧ', '����ҵ��':'ũѧ', '��˾����':'����ѧ', '������̼���':'��ѧ', '���Ľ���':'��ѧ', '��������빤��':'����ѧ', '��֢ҽѧ':'ҽѧ', '�˿ڡ���Դ�뻷������ѧ':'����ѧ', '���õ缼��':'��ѧ', '���ѧ��':'����ѧ', '�������̼�����������':'��ѧ', '�����繤����':'��ѧ', '����':'����ѧ', '�������һ�廯������Ӧ��':'��ѧ', '�����������':'��ѧ', '������������':'����ѧ', '̿�ؼӹ�����':'��ѧ', '������������':'��ѧ', '�ɻ������豸ά��':'��ѧ', 'ʷѧ���ۼ�ʷѧʷ':'��ʷѧ', '���Ľ���':'����ѧ', '������װ':'��ѧ', '�������̼���':'��ѧ', '�ƶ�ͨ�ż���':'��ѧ', 'ú������������':'��ѧ', '�Ź���������������':'��ѧ', 'ˮ��վ�����豸�����':'��ѧ', 'ú�������似��':'��ѧ', '�����ٴ����Ƽ���':'ҽѧ','��������':'����ѧ'

}

def majorClassify(major_):
    m = major_.split('(')[0]
    m = m.strip('��')
    m_ = m[:2]
    if (m_ == '??' or m == '\n'):
        return None
    try:
        first_major = major[m_]
    except:
        first_major = more_major[m]
    return first_major