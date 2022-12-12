import geatpy as ea
import tools
import argparse
import networkx as nx
import numpy as np
import StepOne.FCM as fcm
import Step1Fitness as fit
import Step2Fitness.modularity as modularity
import matplotlib.pyplot as plt


def parse_args(dataset):    # 为dataset名增加路径等参数

    parser = argparse.ArgumentParser(description="Run BiasedWalk.")

    # 子铮学长的拓扑数据集
    parser.add_argument('-graph', nargs='?', default='./Datasets_Attributed Networks/'+dataset + '/network.txt',
                        help='Graph path')
    # 子铮学长的属性数据集
    parser.add_argument('-attributes', nargs='?', default='./Datasets_Attributed Networks/'+dataset + '/features.txt',
                        help='Attributes of nodes')
    # 子铮学长的真实社区划分
    parser.add_argument('-community', nargs='?', default='./Datasets_Attributed Networks/' + dataset + '/communityLabel.txt',
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

def read_network(dataset):
    args = parse_args(dataset)  # 子铮学长数据集命名

    # 读取文件获得整个图
    G = nx.read_edgelist(args.graph, nodetype=int, create_using=nx.DiGraph())
    nodeNum = G.number_of_nodes()
    print("节点数目：",nodeNum)
    # 获得属性特征矩阵
    data = np.loadtxt(args.attributes, dtype=int, encoding='utf-8')
    data = data[np.argsort(data[:, 0])]
    attributeMatrix = np.delete(data, 0, axis=1)
    # print("特征矩阵为：", attributeMatrix)
    return G,attributeMatrix

def run(dataset,clusterNum):
    args = parse_args(dataset)  # 子铮学长数据集命名

    # 读取文件获得整个图
    G = nx.read_edgelist(args.graph, nodetype=int, create_using=nx.DiGraph())
    nodeNum = G.number_of_nodes()
    print(nodeNum)
    # 获得属性特征矩阵
    data = np.loadtxt(args.attributes, dtype=float, delimiter=' ', encoding='utf-8')
    data = data[np.argsort(data[:, 0])]
    attributeMatrix = np.delete(data, 0, axis=1)

    # =================获得真实社区划分===============
    f = open(args.community)
    lines = f.readlines()
    community = []
    for l in lines:
        temp = []
        for node in l.split():
            temp.append(int(node))
        community.append(temp)
    # community = np.loadtxt(args.community, dtype=int, delimiter=' ', encoding='utf-8')
    # print(community)
    print("真实模块度：", modularity.cal_Q(community, G))
    # print("真实划分entropy:", fit.entropy(attributeMatrix.tolist(), args.dimension, community))
    # print("真实划分Dint:", fit.Dint(community, G))
    # print("真实划分Dext:", fit.Dext(community, G))
    # =================暂时不用===============
    # =================暂时不用===============
    # 转化成无向图
    G = G.to_undirected()
    # 获得属性和拓扑的嵌入向量
    # attr_emb_path = './attrWalker/emb/Adw.{:s}.emb'.format("citeseer")
    emb_path = './emb/open-ANE_emb/{:s}_node_embs.emb'.format(dataset)
    embDataTA = np.loadtxt(emb_path, dtype=float, delimiter=' ', skiprows=1)
    # print(embDataTA)
    TnodeID = embDataTA[:, 0].astype(int)
    # print(TnodeID)
    Tvectors = np.delete(embDataTA, 0, axis=1)  # 裁掉第一列的 nodeID
    # print(Tvectors)
    print("=========", clusterNum, "============")
    a = fcm.FCM(Tvectors, clusterNum, 10)
    res = []
    for j in range(clusterNum):
        res.append([])
    for j in range(len(a.label)):
        res[a.label[j]].append(j)
    resFit = fit.Lam(fit.entropy(attributeMatrix.tolist(), args.dimension, res), fit.Dint(res, G),
                          fit.Dext(res, G)) / fit.Sep(clusterNum, a.U)
    # print("KKM_RC", fit.cal_KKM_RC(res, G))
    # print("真实fit：",
    #       fit.Lam(fit.entropy(attributeMatrix.tolist(), args.dimension, community), fit.Dint(community, G),
    #               fit.Dext(community, G)))
    print("模块度：",modularity.cal_Q(res,G))
    return resFit

# print(run("cora",7))