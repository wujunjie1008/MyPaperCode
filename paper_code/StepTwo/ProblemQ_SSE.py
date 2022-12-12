import numpy as np
import geatpy as ea
import Step2Fitness.lxIndex as sim
import argparse
import networkx as nx
import Step2Fitness.modularity as modularity
from scipy.spatial.distance import cdist
def parse_args(dataset):    # 为dataset名增加路径等参数

    parser = argparse.ArgumentParser(description="Run BiasedWalk.")

    # 子铮学长的拓扑数据集
    parser.add_argument('-graph', nargs='?', default='./Datasets_Attributed Networks/'+dataset + '/network.txt',
                        help='Graph path')
    # 子铮学长的属性数据集
    # parser.add_argument('-attributes', nargs='?', default='./Datasets_Attributed Networks/'+dataset + '/features.txt',
    #                     help='Attributes of nodes')
    parser.add_argument('-attributes', nargs='?', default='./Datasets_Attributed Networks/' + dataset + '/remap/'+dataset+'.feat',
                        help='Attributes of nodes')
    # 子铮学长的真实社区划分
    parser.add_argument('-community', nargs='?', default='./Datasets_Attributed Networks/' + dataset + '/community.txt',
                        help='Community')

    # # 林栩拓扑数据集
    # parser.add_argument('-graph', nargs='?', default='./dataset/' + dataset + '/remap/'+ dataset +'.ugraph',
    #                     help='Graph path')
    # # 林栩属性数据集
    # parser.add_argument('-attributes', nargs='?', default='./dataset/' + dataset + '/remap/' + dataset + '.feat',
    #                     help='Attributes of nodes')
    # # 林栩真实数据划分
    # parser.add_argument('-community', nargs='?', default='./dataset/' + dataset + '/remap/' + dataset + '.cmty',
    #                     help='Community')

    # 输出路径
    parser.add_argument('-output', nargs='?', default='emb/' + dataset + '.emb',
                        help='Output path of sparse embeddings')

    # 属性维度
    parser.add_argument('-dimension', type=int, default=1433,
                        help='Number of dimensions. Default is 128.')
    # 随机游走步长
    parser.add_argument('-step', type=int, default=5,
                        help='Step of recursion. Default is 5.')
    return parser.parse_args()


class myProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, dataset,estimateCommunityNum):
        args = parse_args(dataset)  # 子铮学长数据集命名

        # ===============读取文件获得整个图===============
        self.G = nx.read_edgelist(args.graph, nodetype=int, create_using=nx.DiGraph())
        nodeNum = self.G.number_of_nodes()
        # ===============读取文件获得整个图===============

        # ===============获得属性特征矩阵===============
        # data = np.loadtxt(args.attributes, dtype=int, delimiter=' ', encoding='utf-8')
        # data = data[np.argsort(data[:, 0])]
        # attributeMatrix = np.delete(data, 0, axis=1)
        # ===============获得属性特征矩阵===============

        # =================获得嵌入向量===============
        emb_path = './emb/open-ANE_emb/{:s}_node_embs.emb'.format(dataset)
        embDataTA = np.loadtxt(emb_path, dtype=float, delimiter=' ', skiprows=1)
        self.map_node2id = {}
        self.map_id2node = {}
        for i in range(embDataTA.shape[0]):
            node_id = int(embDataTA[i, 0])
            self.map_node2id[node_id] = i
            self.map_id2node[i] = node_id
        TAvectors = np.delete(embDataTA, 0, axis=1)  # 裁掉第一列的 nodeID
        self.TAF = TAvectors   # 计算向量间的相似性
        self.distTA = cdist(self.TAF, self.TAF, metric='sqeuclidean')
        self.S = sim.gen_SimMatrix(self.TAF, 'cosine')
        # =================获得嵌入向量===============

        # =================获得真实社区划分===============
        f = open(args.community)
        lines = f.readlines()
        real_comm = []
        for l in lines:
            temp = []
            for node in l.split():
                temp.append(int(node))
            real_comm.append(temp)
        # print(real_comm)
        print(list(self.G.neighbors(167)))
        k = estimateCommunityNum    # 设置编码位数
        print("真实模块度：", modularity.cal_Q(real_comm, self.G))
        # print("真实SSE：", modularity.cal_Q(real_comm, self.G))
        # =================获得真实社区划分===============

        # 归一化嵌入向量
        self.dists = np.zeros((self.TAF.shape[0], k))
        self.normTAF = np.linalg.norm(self.TAF,axis=1)

        min_node = 0
        max_node = nodeNum - 1

        name = 'findCenter'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        maxormins = [-1,1]   # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）

        Dim = k  # 初始化Dim（决策变量维数） 设置为给定的社区数
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [min_node] * Dim  # 决策变量下界
        ub = [max_node] * Dim # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


    def aimFunc(self, pop): # 目标函数
        x = pop.Phen  # 得到决策变量矩阵,即染色体矩阵
        # print(xx)
        # print(type(x))
        idx = 0
        objV = np.zeros((x.shape[0], 2))
        for xz in x:  # 对于每个排列编码中心点
            # ------
            cen_nodes_num = len(xz)
            distTA = self.distTA[:, xz]
            nearest = np.argmin(distTA, axis=1)  # 最近的中心的索引

            # 收集节点到社区
            comms = [[] for _ in range(cen_nodes_num)]
            comms_id = [[] for _ in range(cen_nodes_num)]
            for i in range(len(nearest)):
                comms[nearest[i]].append(self.map_id2node[i])
                comms_id[nearest[i]].append(i)
            Q = modularity.cal_Q(comms, self.G)
            # ---------------------------属性sse开始
            SSE_TA = 0
            # 根据assignments(comms_id) 计算属性嵌入对应的中心点
            # 然后每个点到当前中心点的距离平方和
            for i in range(cen_nodes_num):
                comm_vec = self.TAF[comms_id[i]]
                TAFVcenter = np.mean(comm_vec, axis=0)
                # print((np.asarray(comm_vec - AFVcenter) ** 2).shape)
                SSE_TA += np.sum(np.sum(np.asarray(comm_vec - TAFVcenter) ** 2, axis=1))
            # ---------------------------属性sse结束
            SSE_TA = cen_nodes_num * SSE_TA
            objV[idx] = [Q, SSE_TA]
            idx += 1
        pop.ObjV = objV