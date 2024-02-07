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
                    GENDER = '女'
                else:
                    GENDER = '男'
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
        self.GDP_embedding = self.gdp_embedding()#(n,) {source_index: gdp_embedding} source index从小到大的顺序排列
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
        sorted_GDP = {}#按照source index从小到大的顺序排列
        for key in sorted(GDP_embedding.keys()):
            sorted_GDP[key] = GDP_embedding[key]
        return sorted_GDP


class HigherDataset(Dataset):
    def __init__(self):
        self.source = []
        self.recipient = []
        self.city = []
        self.province = []
        self.GDP = []#每个学生户籍县的GDP（归一化后）
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

        # 遍历字典，将相邻节点的位置设为1
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
        sorted_GDP = {}#按照source index从小到大的顺序排列
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
more_major = {'飞机机电设备维修':'工学', '渔业':'农学', '管道工程技术':'工学', '民间文学':'文学', '初等教育学':'教育学', '室内装饰设计':'艺术学', '对外汉语教学':'教育学', '清洁生产与减排技术':'工学', '煤化工技术':'工学', '地方政府学':'管理学', '风能发电设备制造与维修':'工学', '寒区工程与环境':'工学', '现代纺织技术':'工学', '工科试验班':'工学', '农产品安全与环境':'农学', '核能科学与工程':'工学', '兽药制药技术':'医学', '媒介经营与管理':'管理学', '橡胶工程技术':'工学', '文艺与传媒':'文学', '投融资管理':'管理学', '媒介与社会变迁':'法学', '儿童文学':'文学', '林业':'农学', '区域环境学':'工学', '舞台艺术设计与制作':'艺术学', '模特与礼仪':'艺术学', '市政工程技术':'工学', '作物栽培学与耕作学':'农学', '数控设备应用与维护':'工学', '建设工程监理':'工学', '矿产普查与勘探':'理学', '可再生能源科学与工程':'工学', '公路工程造价管理':'工学', '研究生教育学':'教育学', '水环境监测与保护':'工学', '制冷与空调技术':'工学', '先进高分子材料':'工学', '餐饮管理':'管理学', '交互媒体设计':'艺术学', '景观规划设计':'艺术学', '核技术与材料工程':'工学', '火炮、自动武器与弹药工程':'工学', '油田化学应用技术':'工学', '亚非语言文学':'文学', '人物形象设计':'艺术学', '理化测试与质检技术':'工学', '民航运输':'管理学', '法律':'法学', '刺绣设计与工艺':'艺术学', '电厂设备运行与维护':'工学', '棉花加工与经营管理':'管理学', '藏学':'文学', '法律治理学':'法学', '移动应用开发':'工学', '皮具制作与工艺':'艺术学', '地图学与地理信息系统':'理学', '药品制造类':'医学', '钻井技术':'工学', '机场场务技术与管理':'管理学', '皮具设计':'艺术学', '中小企业创业与经营':'管理学',
             '跨国供应链管理':'管理学', '元培计划实验班':'教育学', '结构工程':'工学', '游艇设计与制造':'工学', '杂草学':'农学', '权籍信息化管理':'管理学', '农产品质量与食物安全':'农学', '船机制造与维修':'工学', '飞机控制设备与仪表':'工学', '作物学':'农学', '文学与翻译':'文学', '家用纺织品设计':'艺术学', '构造地质学':'理学', '内科学':'医学', '传媒策划与管理':'管理学', '地图制图与数字传播技术':'工学', '宝玉石鉴定与加工':'工学', '人体生理学':'理学', '文秘':'管理学', '新型建筑材料技术':'工学', '矿山测量':'工学', '轻工类':'工学', '针织技术与针织服装':'工学', '农产品质量检测':'农学', '云计算技术与应用':'工学', '发展经济学':'经济学', '电力系统及其自动化':'工学', '水资源利用与保护':'工学', '激光加工技术':'工学', '音像技术':'艺术学', '设计学':'艺术学', '燃烧动力学':'工学', '热带园艺作物学':'农学', '固体力学':'工学', '太阳能光热技术与应用':'工学', '检测技术及应用':'工学', '刑法学':'法学', '国外马克思主义研究':'哲学', '无线电物理':'工学', '高速铁路客运乘务':'管理学', '青少年工作与管理':'管理学', '非金属矿物材料技术':'工学', '林业经济管理':'管理学', '药代动力学':'医学', '健行学院实验班类':'教育学', '电工理论与新技术':'工学', '天体物理':'理学', '光伏发电技术与应用':'工学', '可再生洁净能源':'工学', '飞机结构修理':'工学', '言语听觉康复科学':'医学', '展示艺术设计':'艺术学', '水电站动力设备':'工学', '核技术及应用':'工学', '传媒法学':'法学', '传媒经济学':'经济学', '法律文秘':'法学', '捕捞学':'农学', '草地生物多样性':'农学', '水工结构工程':'工学', '先进制造':'工学', '电厂化学':'工学', '有色冶金技术':'工学',
             '细胞生物学':'理学', '畜产品加工工程':'工学', '农产品质量安全':'农学', '基因组学':'理学', '微尺度科学与技术':'工学', '学科教':'教育学', '医药营销':'管理学', '楼宇智能化工程技术':'工学', '外科学':'医学', '发电厂及电力系统':'工学', '养殖':'农学', '动漫设计与制作':'艺术学', '民俗学':'文学', '畜牧工程技术':'工学', '钢铁冶金设备应用技术':'工学', '西餐工艺':'艺术学', '药用植物资源工程':'农学', '写作学':'文学', '版面编辑与校对':'文学', '儿童口腔医学':'医学', '比较教育学':'教育学', '畜牧业经济管理':'经济学', '染整技术':'工学', '矿井通风与安全':'工学', '农产品加工及贮藏工程':'工学', '矿业工程':'工学', '区域文化学':'文学', '宝玉石鉴定与加工技术':'工学', '一般力学与力学基础':'理学', '学科教学':'教育学', '岩矿分析与鉴定':'工学', '内燃机制造与维修':'工学', '现代农业技术':'农学', '国土资源调查与管理':'管理学', '保护生物学':'理学', '中共党史':'历史学', '比较政治':'法学', '比较文学与世界文学':'文学', '作物生产技术':'农学', '水信息学':'理学', '民政管理':'管理学', '微生物学':'理学', '饭店管理':'管理学', '分布式发电与微电网技术':'工学', '光通信技术':'工学', '军队指挥学':'法学', '德育学':'教育学', '大地测量学与测量工程':'工学', '轻工技术与工程':'工学', '兵器工程':'工学', '粮油储藏与检测技术':'农学', '第四纪地质学':'理学', '眼科学':'医学', '光纤通信':'工学', '企业诊断学':'管理学', '风电系统运行与维护':'工学', '比较制度学':'法学', '检测技术与自动化装置':'工学', '种业':'农学', '课程与教学论':'教育学', '电梯工程技术':'工学', '药品生产技术':'医学', '室内艺术设计':'艺术学', '干细胞和再生医学':'医学', '电源变换技术与应用':'工学', '高电压与绝缘技术':'工学',
              '建设工程管理类':'管理学', '儿少卫生与妇幼保健学':'医学', '装饰艺术设计':'艺术学', '企业经济学':'经济学', '光学工程':'工学', '婚庆服务与管理':'管理学', '水电站机电设备与自动化':'工学', '耳鼻咽喉科学':'医学', '特警':'法学', '民事执行':'法学', '土建施工类':'工学', '生药学':'医学', '区域发展与城乡规划':'管理学', '生化制药技术':'医学', '农产品加工与质量检测':'农学', '报关与国际货运':'管理学', '建设项目信息化管理':'管理学', '幼儿发展与健康管理':'教育学', '国防经济':'经济学', '西北地区对外开放法律问题研究':'法学', '电力系统继电保护与自动化技术':'工学', '师范类':'教育学', '热能动力设备与应用':'工学', '天体测量与天体力学':'理学', '高压输配电线路施工运行与维护':'工学', '政府采购管理':'管理学', '高速动车组驾驶':'工学', '转轨经济学':'经济学', '药用植物学':'农学', '信号与信息处理':'工学', '新型纺织机电技术':'工学', '少数民族传统体育':'教育学', '人体解剖与组织胚胎学':'医学', '生产过程自动化技术':'工学', '供热通风与空调工程技术':'工学', '岩土力学与工程':'工学', '医药信息管理':'管理学', '导弹维修':'工学', '企业管理':'管理学', '制冷及低温工程':'工学', '青年与国际政治':'法学', '风能与动力技术':'工学', '药用植物资源':'农学', '动力工程':'工学',
              '编导':'艺术学', '税务':'经济学', '主持与播音':'艺术学', '发酵工程':'工学', '电力系统自动化技术':'工学', '畜牧':'农学', '饲草生产加工及利用':'农学', '林业工程类':'工学', '生命信息物理学':'理学', '雷电科学与技术':'工学', '固体电子物理':'理学', '人权法学':'法学', '渔业发展':'农学', '作物遗传育种':'农学', '文艺学':'文学', '歌舞表演':'艺术学', '国土测绘与规划':'工学', '直升机驾驶技术':'工学', '有机化学':'理学', '景观生态规划与管理':'工学', '仪器仪表工程':'工学', '生理生态学':'理学', '现代殡葬技术与管理':'管理学', '文科试验班类':'教育学', '爆破理论与技术':'工学', '地学信息工程':'工学', '中外语言文化比较研究':'文学', '少年儿童组织与思想意识发展':'教育学', '力学类':'理学', '饲草学':'农学', '人口学':'经济学', '再生生物学':'理学', '林业工程':'工学', '光通信与光传感':'工学', '民航通信技术':'工学', '高尔夫球运动与管理':'管理学', '航天工程':'工学', '网站规划与开发技术':'工学', '矿井建设':'工学', '古建筑工程技术':'工学', '铸造技术':'工学', '警察管理':'管理学', '教师发展与管理':'教育学', '出版商务':'管理学', '光机电工程与应用':'工学', '高速铁道工程技术':'工学', '数控技术':'工学', '宪法学与行政法学':'法学', '皮肤病与性病学':'医学', '光子学与光子技术':'工学', '天然产物化学':'理学', '计量心理学':'理学', '工科试验班类':'教育学', '药剂学':'医学', '制浆造纸技术':'工学', '品牌代理经营':'管理学', '营养与食品卫生学':'医学', '林业信息技术与管理':'管理学', '水政水资源管理':'管理学', '表面精饰工艺':'工学', '草学':'农学',
              '欧洲学':'历史学', '法律经济学':'经济学', '非洲学':'历史学', '电脑艺术设计':'艺术学', '图形图像制作':'艺术学', '区域经济学':'经济学', '精准医学与公共健康':'医学', '综合文科教育':'教育学', '比较文学与文化理论':'文学', '器乐演奏与教学研究':'艺术学', '药品生物技术':'医学', '入侵生物学':'理学', '节电技术与管理':'管理学', '硅材料制备技术':'工学', '林特食品加工与开发':'农学', '医药生物学':'医学', '水电站运行与管理':'工学', '高压科学与技术':'工学', '首饰设计与工艺':'艺术学', '发展与教育心理学':'教育学', '营养配餐':'管理学', '兽药学':'农学', '畜牧学':'农学', '煤炭深加工与利用':'工学', '光伏发电技术及应用':'工学', '有色金属冶金':'工学', '中外政治制度':'哲学', '电厂化学与环保技术':'工学', '政府绩效管理':'管理学', '土壤学':'农学', '桥梁与隧道工程':'工学', '电力技术类':'工学', '药用植物资源与利用':'农学', '绿色能源化学与技术':'工学', '通用航空航务技术':'工学', '雕刻艺术设计':'艺术学', '乐器制造与维护':'艺术学', '光伏材料加工与应用技术':'工学', '天然药物化学':'医学', '母婴医学':'医学', '结构生物学':'理学', '景区开发与管理':'管理学', '警察科学':'管理学', '妇产科学':'医学', '文体研究与文学教育':'文学', '原子与分子物理':'理学', '高等教育学':'教育学', '动车组检修技术':'工学', '法律事务':'法学', '矿业贸易与投资':'经济学', '皮革化学与工程':'工学', '水电站与电力网':'工学', '再生医学':'医学', '营养与食品安全':'管理学', '转化医学':'医学', '区域管理与公共政策':'管理学', '戏曲表演':'艺术学',
              '宠物养护与疫病防治':'农学', '作物生物技术':'农学', '测试计量技术及仪器':'工学', '综合机械化采煤':'工学', '游戏软件':'艺术学', '电视节目制作':'艺术学', '干细胞与再生医学':'医学', '村镇建设与管理':'管理学', '风力发电工程技术':'工学', '水力学及河流动力学':'理学', '理科试验班类':'教育学', '出版与电脑编辑技术':'文学', '非常规油气地质学':'理学', '港航物流类':'管理学', '法律史':'法学', '初等教育':'教育学', '林业[设林业':'农学', '展示设计':'艺术学', '华语与华文教育':'教育学', '矿业类':'工学', '成人教育学':'教育学', '林区建筑与结构工程':'工学', '高速动车组检修技术':'工学', '可再生能源与清洁能源':'工学', '应激生物学':'理学', '光通信与光信息技术':'工学', '全科医学':'医学', '矿山地质':'工学', '水路运输与海事管理':'管理学', '警察指挥与战术':'管理学', '公路监理':'工学', '粒子物理与原子核物理':'理学', '矿山机电技术':'工学', '教师教育':'教育学', '盾构施工技术':'工学', '媒体营销':'管理学', '制造业信息化技术':'工学', '种质资源保护与利用':'农学', '选煤技术':'工学', '情报学':'管理学', '高尔夫服务与管理':'管理学', '核燃料循环与材料':'工学', '仪器科学与技术':'工学', '草地营养生物学':'农学', '湘绣设计与工艺':'艺术学', '绿色化学与技术':'工学', '飞机制造技术':'工学', '兵器科学与技术':'工学', '热工检测及控制技术':'工学', '作物':'农学', '有机化工生产技术':'工学', '轻工纺织食品类':'工学', '集装箱运输管理':'管理学', '舆论学':'文学', '宠物养护与驯导':'艺术学', '矿山空间信息工程':'工学', '病理学与病理生理学':'医学', '壮学与瑶学':'历史学', '比较医学':'医学', '文秘速录':'文学', '绿色包装与低碳管理':'管理学', '出版':'文学', '城镇建设':'工学', '多媒体设计与制作':'艺术学',
              '制浆造纸工程':'工学', '室内设计技术':'艺术学', '室内环境检测与控制技术':'工学', '洁净能源科学':'理学', '制糖生产技术与管理':'工学', '烹调工艺与营养':'艺术学', '戒毒矫治技术':'医学', '移动商务':'管理学', '木基复合材料科学与工程':'工学', '同步辐射及应用':'理学', '方剂学':'医学', '输血医学':'医学', '少年儿童组织与思想意识教育':'教育学', '游戏设计与制作':'艺术学', '给水排水工程':'工学', '非常规油气工程':'工学', '政府经济学':'经济学', '边疆地理学':'理学', '污染修复与生态工程技术':'工学', '分析化学':'理学', '急诊医学':'医学', '控制理论与控制工程':'工学', '钢琴调律':'艺术学', '煤化分析与检验':'工学', '免疫学':'医学', '药品质量与安全':'医学', '统一战线学':'哲学', '黑色冶金技术':'工学', '水环境监测与治理':'工学', '有机硅化学及材料':'工学', '输电工程':'工学', '鞋类设计与工艺':'艺术学', '通用航空器维修':'工学', '民航商务':'管理学', '渔业资源':'农学', '高尔夫运动技术与管理':'管理学', '防雷技术':'工学', '水上运输类':'工学', '运筹学与控制论':'工学', '会计学':'经济学','作物安全生产':'农学', '西北地区民族宗教法律问题研究':'法学', '连锁经营管理':'管理学', '边疆学':'历史学', '企业文化与伦理':'哲学', '传媒社会学':'文学', '维药学':'理学', '证券与期货':'经济学', '遗传学':'理学', '仪器类':'工学', '民航空中安全保卫':'工学', '核天体物理':'理学', '烟花爆竹技术与管理':'工学', '观赏园艺':'农学', '绿色化学过程与资源综合利用':'工学', '创意产业与社会管理':'管理学', '绿色化学':'工学', '本草生物学':'理学', '矿业装备维护技术':'工学', '绿色食品生产与检验':'农学', '农畜特产品加工':'农学', '市政工程':'工学', '热能与发电工程类':'工学', '肿瘤学':'医学', '畜牧兽医':'农学',
              '审美文化与文学':'文学', '核能与核技术工程':'工学', '法律执行类':'法学', '矿产地质与勘查':'工学', '电线电缆制造技术':'工学', '机动车保险实务':'法学', '图文信息处理':'工学', '制冷与冷藏技术':'工学', '农产品安全':'农学', '公路运输与管理':'管理学', '互动艺术与技术':'艺术学', '诉讼法学':'法学', '固体地球物理学':'理学', '敦煌学':'历史学', '电力工程经济与管理':'管理学', '合作经济学':'经济学','热能工程':'工学', '电厂热能动力装置':'工学', '移动互联网应用技术':'工学', '蔬菜学':'农学', '皮具艺术设计':'艺术学', '游戏设计':'艺术学', '产业经济学':'经济学', '对外汉语':'文学', '港航技术与管理工程':'工学', '保健品开发与管理':'管理学', '病原生物学':'理学', '防水材料与工程':'工学', '区域发展与管理创新':'管理学', '公司财务战略与管理':'管理学', '兽药生产与营销':'管理学', '流通经济学':'经济学', '营销管理':'管理学', '药品质量检测技术':'理学', '观赏园艺学':'农学', '餐饮食品安全管理':'管理学', '风力发电设备及电网自动化':'工学', '火电厂集控运行':'工学', '林业技术':'农学', '比较文学与跨文化研究':'文学', '营养与配餐':'理学', '商检技术':'管理学', '供热、供燃气、通风及空调工程':'工学', '轻工装备及控制':'工学', '水灾害与水安全':'工学', '文献资源保护与利用':'管理学', '茶叶生产加工技术':'农学', '机场运行':'工学', '非常规油气地质与勘探':'工学', '嵌入式技术与应用':'工学', '电力系统继电保护与自动化':'工学', '数量经济学':'经济学', '公路工程检测技术':'工学', '动漫设计':'艺术学', '旅游管理':'管理学', '水利水电工程':'工学',
            '彝学':'历史学', '设计艺术学':'艺术学', '畜产品加工与安全':'农学', '建设工程管理':'管理学', '高等级公路维护与管理':'工学', '凝聚态物理':'理学', '水信息技术':'工学', '集散控制系统应用与维护技术':'工学', '证券投资与管理':'经济学', '电厂热工自动化技术':'工学', '装潢艺术设计':'艺术学', '罪犯心理测量与矫正技术':'理学', '制糖工程':'工学', '模式识别与智能系统':'工学', '快递运营管理':'管理学', '美学':'哲学', '跨文化交际与翻译':'文学', '法行政学':'法学', '生理学':'理学', '力学':'理学', '煤田地质与勘查技术':'工学', '光学':'理学', '农资营销与服务':'农学', '选矿技术':'工学', '能效工程':'工学', '飞机维修':'工学', '茶艺与茶叶营销':'农学', '定翼机驾驶技术':'工学', '岩土工程':'工学', '呼吸治疗技术':'医学', '腐蚀科学与防护':'工学', '专门史':'历史学', '戏曲':'艺术学', '电梯维护与管理':'工学', '舞台影视技术':'艺术学', '流体机械及工程':'工学', '救援技术':'工学', '法律实务类':'法学', '煤矿开采技术':'工学', '消费经济学':'经济学', '法律硕士':'法学', '通用航空飞行器设计与制造':'工学', '恢复生态学':'理学', '核科学与技术':'工学','牧区地理学':'理学', '车身工程':'工学', '理科试验班':'教育学', '荒漠生态学':'理学', '钢铁冶金':'工学', '工学大类':'工学', '城镇规划':'理学', '兵器类':'工学', '轧钢工程技术':'工学', '台湾研究':'历史学', '控制科学与工程':'工学', '餐饮类':'管理学', '健身指导与管理':'管理学', '地籍测绘与土地管理信息技术':'工学', '3S集成与气象应用':'工学', '传媒艺术学':'艺术学', '高速铁路动车乘务':'管理学', '感染病学':'医学', '花卉与景观园艺':'农学', '发酵技术':'工学', '果树学':'农学', '媒介文化传播':'文学', '电力信息技术':'工学',
              '言语听觉康复技术':'医学', '渔业经济管理':'管理学', '煤化工':'工学', '雕刻艺术与家具设计':'艺术学', '防火管理':'管理学', '冷链物流技术与管理':'管理学', '牧草育种与栽培':'农学', '病理生理学':'医学', '天体物理及天文数据信息处理':'理学', '先进材料与制备技术':'工学', '早期教育':'教育学', '钢结构建造技术':'工学', '作物种质资源学':'农学', '电路与系统':'工学', '兵器发射理论与技术':'工学', '高等语文教育':'教育学', '液压与气动技术':'工学', '生殖医学':'医学', '微生物与生化药学':'医学', '光伏材料制备技术':'工学', '沉积学':'理学', '钻探技术':'工学', '有机电子学':'工学', '农产品流通与管理':'管理学', '模具设计与制造':'工学', '电力客户服务与管理':'管理学', '现代流行音乐':'艺术学', '医药卫生法学':'法学', '公路机械化施工技术':'工学', '林业信息工程与管理':'管理学', '西方经济学':'经济学', '旅行社经营管理':'管理学', '流体力学':'工学', '飞机部件修理':'工学', '民航安全技术管理':'管理学', '畜牧生物工程':'农学', '作物生理学':'农学', '概率论与数理统计':'理学', '国学':'哲学', '营养代谢免疫学':'医学', '影像医学与核医学':'医学', '药品经营与管理':'管理学', '草原学':'农学', '控制工程':'工学', '现代教育技术':'教育学', '药理学':'医学', '营销与策划':'管理学', '餐饮管理与服务':'管理学', '人机与环境工程':'工学', '发育生物学':'理学', '声像工程技术':'工学', '循证医学':'医学', '高速铁道技术':'工学', '比较法学':'法学', '药品服务与管理':'管理学', '无损检测技术':'工学', '茶树栽培与茶叶加工':'农学', '林业调查与信息处理':'工学', '藏语信息处理工程':'工学', '地籍测绘与土地管理':'管理学', '市政工程类':'工学', '宏观质量管理':'管理学', '公路环境工程':'工学', '河口海岸学':'理学',
              '海岸带资源与环境':'理学', '矿产勘查与地质环境':'理学', '时间生物学':'理学', '载运工具运用工程':'工学', '法制新闻与传媒法':'法学', '修复生态学':'理学', '移动互联应用技术':'工学', '动力机械及工程':'工学', '钢琴伴奏':'艺术学', '渔业经济与管理':'管理学', '藏语言文学':'文学', '林木遗传育种':'农学', '矿山机电':'工学', '动漫制作技术':'艺术学', '有害生物与环境安全':'理学', '公益慈善事业管理':'管理学', '地图制图学与地理信息工程':'工学', '检察事务':'法学', '电力电子与电力传动':'工学', '太阳能光热技术及应用':'工学', '民航特种车辆维修':'工学', '空中乘务':'管理学', '民商法学':'法学', '畜牧业类':'农学', '公司治理':'管理学', '光伏工程技术':'工学', '党的建设':'哲学', '玉器设计与工艺':'艺术学', '重症医学':'医学', '人口、资源与环境经济学':'经济学', '供用电技术':'工学', '设计学类':'艺术学', '动力工程及工程热物理':'工学', '机场电工技术':'工学', '导游':'管理学', '光伏建筑一体化技术与应用':'工学', '领域软件工程':'工学', '美容美体艺术':'艺术学', '炭素加工技术':'工学', '等离子体物理':'理学', '飞机电子设备维修':'工学', '史学理论及史学史':'历史学', '语文教育':'教育学', '人民武装':'哲学', '岩土工程技术':'工学', '移动通信技术':'工学', '煤化工生产技术':'工学', '磁光电材料物性与器件':'工学', '水电站动力设备与管理':'工学', '煤层气采输技术':'工学', '宠物临床诊疗技术':'医学','合作经济':'经济学'

}

def majorClassify(major_):
    m = major_.split('(')[0]
    m = m.strip('★')
    m_ = m[:2]
    if (m_ == '??' or m == '\n'):
        return None
    try:
        first_major = major[m_]
    except:
        first_major = more_major[m]
    return first_major