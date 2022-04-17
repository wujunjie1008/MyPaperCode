import geatpy as ea
import tools
import argparse
import networkx as nx
import numpy as np


def parse_args(dataset):    # 为dataset名增加路径等参数

    parser = argparse.ArgumentParser(description="Run BiasedWalk.")
    # 子铮学长的拓扑数据集
    # parser.add_argument('-graph', nargs='?', default='./Datasets_Attributed Networks/'+dataset + '/network.txt',
    #                     help='Graph path')
    # 子铮学长的属性数据集
    # parser.add_argument('-attributes', nargs='?', default='./Datasets_Attributed Networks/'+dataset + '/features.txt',
    #                     help='Attributes of nodes')

    # 林栩拓扑数据集
    parser.add_argument('-graph', nargs='?', default='./dataset/' + dataset + '/remap/'+ dataset +'.ugraph',
                        help='Graph path')
    # 林栩属性数据集
    parser.add_argument('-attributes', nargs='?', default='./dataset/' + dataset + '/remap/' + dataset + '.feat',
                        help='Attributes of nodes')
    # 输出路径
    parser.add_argument('-output', nargs='?', default='emb/' + dataset + '.emb',
                        help='Output path of sparse embeddings')
    # 属性维度
    parser.add_argument('-dimension', type=int, default=128,
                        help='Number of dimensions. Default is 128.')
    # 随机游走步长
    parser.add_argument('-step', type=int, default=5,
                        help='Step of recursion. Default is 5.')
    return parser.parse_args()

# args = parse_args("CiteSeer") # 子铮学长数据集命名
args = parse_args("citeseer")   # 林栩数据集命名

# 读取文件获得整个图
G = nx.read_edgelist(args.graph, nodetype=int, create_using=nx.DiGraph())
# 转化成无向图
G = G.to_undirected()

print(G.number_of_nodes())

# 获得属性和拓扑的嵌入向量
attr_emb_path = './attrWalker/emb/Adw.{:s}.emb'.format("citeseer")
topo_emb_path = './emb/dw.{:s}.emb'.format("citeseer")
data = np.loadtxt(attr_emb_path, dtype=float, delimiter=' ', skiprows=1)
AnodeID = data[:, 0].astype(int)
Avectors = np.delete(data, 0, axis=1)  # 裁掉第一列的 nodeID
print(Avectors.shape)
data = np.loadtxt(topo_emb_path, dtype=float, delimiter=' ', skiprows=1)
TnodeID = data[:, 0].astype(int)
Tvectors = np.delete(data, 0, axis=1)  # 裁掉第一列的 nodeID



