import random
import numpy as np
import networkx as nx
import ReadData as rd
import Locus_base.LocusBase_tools as locus_base

def init(num,dataset,threshold):
    G, attributeMatrix = rd.read_network(dataset)
    nodeNum = G.number_of_nodes()
    res = []
    attr_cos = []
    for i in range(nodeNum):
        attr_cos.append([])
        for j in range(nodeNum):
            if(i==j):
                attr_cos[i].append(0)
            else:
                attr_cos[i].append(attributeMatrix[i].dot(attributeMatrix[j])/(np.linalg.norm(attributeMatrix[i]) * np.linalg.norm(attributeMatrix[j])))
    for i in range(num):
        res.append([])
        for j in range(nodeNum):
            # 获取节点j的所有邻居节点
            j_neighbors_topo = list(nx.all_neighbors(G, j))
            # 获取节点j的所有属性上余弦相似度大于阈值的点
            j_neighbors_attr = []
            for k in range(nodeNum):
                if attr_cos[j][k]>threshold:
                    j_neighbors_attr.append(k)
            j_neighbors = j_neighbors_topo+j_neighbors_attr
            # 随机挑选其中一个作为locus-base编码中第j位的值
            res[i].append(j_neighbors[random.randint(0, len(j_neighbors)-1)])
    return res,attr_cos

init(3, "texas",0.45)