import networkx as nx
import numpy as np
import math
from metric import modularity
from scipy.spatial.distance import cdist

def cal_Q(G, community):
    m = G.number_of_edges()
    Qsum = 0
    for c in community:
        sum = 0
        for i in c:
            for j in c:
                # print(i,j)
                if G.has_edge(i,j):
                    sum += 1 - G.degree(i) * G.degree(j) / (2*m)
                    # sum += 1
                else:
                    sum += 0 - G.degree(i) * G.degree(j) / (2*m)
                    # sum += 0
        print(c, sum)
        Qsum += sum
    print(Qsum / (2 * m))

def cal_Q_M(G,mem_matrix):
    """
    calculate cosine similarity using member matrix
    :param community:
    :param node_features:
    :return:
    """
    m = G.number_of_edges()
    n = G.number_of_nodes()
    A = np.zeros((n,n))
    # A = np.array(nx.adjacency_matrix(G).todense())
    for u,v in G.edges():
        u -= 1
        v -= 1
        A[u][v] = 1
        A[v][u] = 1

    D = np.array(A.sum(1)).reshape(-1,1)
    Dij = D.dot(D.T)
    B = A - Dij / (2 * m)
    # B = A
    H = mem_matrix
    Q = np.trace(H.T.dot(B).dot(H))/ (2 * m)
    # print(Q)
    return Q

def cal_simCos_M(mem_matrix, feat_matrix):
    """
    calculate cosine similarity using member matrix
    :param community:
    :param node_features:
    :return:
    """
    # print(1)
    A_inner = feat_matrix.dot(feat_matrix.T) # 相似度矩阵
    A_inner = A_inner - np.diag(A_inner.diagonal()) # 去对角元素
    A_norm = np.linalg.norm(feat_matrix,axis=1) # 计算每个向量的范数
    A_norm = A_norm.reshape((A_norm.shape[0],-1))
    A_ij_norm = A_norm.dot(A_norm.T) # 两两范数相乘

    # print(2)
    #
    k = mem_matrix.shape[1]
    # B = A_inner / A_ij_norm # 得到变换矩阵
    B = np.divide(A_inner, A_ij_norm, out=np.zeros_like(A_ij_norm), where=A_ij_norm != 0)
    H = mem_matrix
    #
    # print(3)
    # # 社区如果存在孤立点 (memSizeArr - 1) = 0 就会导致 norm_matrix = 0
    memSizeArray = np.array(mem_matrix.T.sum(1))
    memSizeDiag = np.diag(memSizeArray * (memSizeArray - 1) - 1)
    ones = np.ones_like(memSizeDiag)
    norm_matrix = ones + memSizeDiag
    #
    # print(4)
    # norm_matrix[norm_matrix == 0] = 1

    # # 如果使用归一化矩阵只需要除以 k. 因为归一化矩阵中边就统计了两次，消掉
    norm_M = np.divide(H.T.dot(B).dot(H), norm_matrix, out=np.zeros_like(norm_matrix), where=norm_matrix != 0)
    simCos = np.trace(norm_M) / k
    # simCos = np.trace(H.T.dot(B).dot(H) / norm_matrix) /  k
    # simCos = np.trace(H.T.dot(B).dot(H))/ (2 * k)

    return simCos

def cal_simHaiming_M(mem_matrix, feat_matrix):
    X = feat_matrix
    dist = cdist(X, X, metric='hamming')
    # B = 1 / (1 + dist)
    # print(X.shape[1])
    # dist = dist * X.shape[1]
    # print(dist[1][1])
    # print(dist[1][0])
    # print(dist[1][2])
    # exit()
    # B = dist / X.shape[1]

    B = 1 - dist
    B = B - np.diag(B.diagonal())  # 去对角元素

    k = mem_matrix.shape[1]
    H = mem_matrix


    memSizeArray = np.array(mem_matrix.T.sum(1))
    memSizeDiag = np.diag(memSizeArray * (memSizeArray - 1) - 1)
    ones = np.ones_like(memSizeDiag)
    norm_matrix = ones + memSizeDiag

    norm_M = np.divide(H.T.dot(B).dot(H), norm_matrix, out=np.zeros_like(norm_matrix), where=norm_matrix != 0)
    simHM = np.trace(norm_M)/ k

    return simHM

def cal_simED_M(mem_matrix, feat_matrix, sigma = 5):
    """

    :param G:
    :param mem_matrix:
    :param feat_matrix:
    :param sigma: this parameter required to be adjusted according to attributes
    :return:
    """
    X = feat_matrix
    # dist = cdist(X,X,metric = 'euclidean')
    dist = cdist(X, X, metric='sqeuclidean') # square of euclidean

    # B = 1 / (1 + dist)
    # B = dist

    # K(i, j) = exp(- | x_i - x_j | ^ 2 / (2 * sigma) ^ 2)
    B = np.exp(- dist / (2 * sigma * sigma) )


    B = B - np.diag(B.diagonal())  # 去对角元素

    # print(dist)
    k = mem_matrix.shape[1]
    H = mem_matrix

    memSizeArray = np.array(mem_matrix.T.sum(1))
    memSizeDiag = np.diag(memSizeArray * (memSizeArray - 1) - 1)
    ones = np.ones_like(memSizeDiag)
    norm_matrix = ones + memSizeDiag

    norm_M = np.divide(H.T.dot(B).dot(H), norm_matrix, out=np.zeros_like(norm_matrix), where=norm_matrix != 0)
    simED = np.trace(norm_M)/ k
    # print(simED)
    return simED

def cal_simJac_M(mem_matrix, feat_matrix):
    X = feat_matrix
    dist = cdist(X,X,metric = 'jaccard')
    dist = 1 - dist
    B = dist - np.diag(dist.diagonal())  # 去对角元素

    k = mem_matrix.shape[1]
    H = mem_matrix

    memSizeArray = np.array(mem_matrix.T.sum(1))
    memSizeDiag = np.diag(memSizeArray * (memSizeArray - 1) - 1)
    ones = np.ones_like(memSizeDiag)
    norm_matrix = ones + memSizeDiag

    simJac = np.trace(H.T.dot(B).dot(H) / norm_matrix)/ k
    # print(simJac)

    return simJac


def cal_simED(community, node_features):
    # i < j
    k = len(community)
    simED = 0

    for c in community:
        comSim = 0
        for i in range(len(c) - 1):
            for j in range(i + 1, len(c)):
                a = np.array(node_features[c[i]])
                b = np.array(node_features[c[j]])
                d = np.linalg.norm (a - b)
                d = 1 / (1 + d)
                # normalize <- d_norm = 1 / ( 1 + d)
                # or d_norm = exp (-d/Z)
                # or RBF K(i,j) = exp(-|x_i-x_j| ^ 2)
                comSim += d
        # simED += comSim
        simED += 2 * comSim / (len(c) * (len (c) - 1))

    return simED / k

def cal_simCos(community, node_features):
    # i < j
    k = len(community)
    simCos = 0
    for c in community:
        comSim = 0
        for i in range(len(c) - 1):
            for j in range(i + 1, len(c)):
                a = np.array(node_features[c[i]])
                b = np.array(node_features[c[j]])
                nume = np.inner(a,b)
                denomi = np.linalg.norm(a) * np.linalg.norm(b)
                cos_sim = nume / denomi
                # normalize <- cos_sim = 0.5 + 0.5 * cos_sim
                comSim += cos_sim

        # simCos += comSim
        simCos += 2 * comSim / (len(c) * (len (c) - 1))


    return simCos / k

def cal_simJaccard(community, node_features):
    # i < j
    k = len(community)
    features_sum = len(list(node_features.values())[0])
    simJacc = 0
    for c in community:
        comSim = 0
        for i in range(len(c) - 1):
            for j in range(i + 1, len(c)):
                intersection = 0
                union = 0
                for f in range(features_sum):
                    a = node_features[c[i]][f] + node_features[c[j]][f]
                    if a > 0 : union += 1
                    if a > 1 : intersection += 1
                comSim += intersection / union

        # simJacc += comSim
        simJacc += 2 * comSim / (len(c) * (len (c) - 1))

    # print(k)
    return simJacc / k

# 计算社区链接密度。
def cal_density(G, community, node_neighbors):
    density = 0
    edges_sum = len(G.edges())
    for c in community:
        cnt = 0
        for i in range(len(c) - 1):
            for j in range(i + 1, len(c)):
                if c[j] in node_neighbors[c[i]]:
                    cnt += 1

        d = cnt / edges_sum
        density += d

    return density


# 计算社区属性信息熵。
def cal_entropy(node_sum, community, node_features):
    entropy = 0
    features_sum = len(list(node_features.values())[0])
    community_sum = len(community)

    # for ci in community:
    #     for j in range(features_sum):
    #         cnt = 0
    #         for n in ci:
    #             if node_features[n][j] == 1: cnt+= 1
    #         pij = cnt / len (ci)
    #
    #         if pij != 0:
    #             entropy_aj_ci = -pij * (math.log(pij, 2))
    #             # print(pij)
    #         else:
    #             # 社区中的点都没有这个属性时置零。
    #             entropy_aj_ci = 0
    #
    #         entropy += ((len(ci) / node_sum) * entropy_aj_ci)

    for i in range(features_sum):
        for j in range(community_sum):
            nodes_in_cj = community[j]
            cj_size = len(nodes_in_cj)

            cnt = 0
            for n in nodes_in_cj:
                n_features = node_features[n]
                if n_features[i] == 1: cnt += 1
            p_ij = cnt / cj_size
            if p_ij != 0:
                entropy_ai_cj = -p_ij * (math.log(p_ij, 2))
            else:
                # 社区中的点都没有这个属性时置零。
                entropy_ai_cj = 0
            entropy += ((cj_size / node_sum) * entropy_ai_cj)

    return entropy


# 载入网络，社区，属性并处理成需要的数据。
def load_data(net_file, com_file, feat_file):
    G = nx.read_edgelist(net_file,nodetype=int)
    node_neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}

    community = []
    with open(com_file, 'r') as f:
        for row in f:
            nodes = list(map(int,row.strip().split()))
            community.append(nodes)

    raw_node_features = np.genfromtxt(feat_file, dtype=np.dtype(str))
    features = np.array(raw_node_features[:, 1:], dtype=np.int32)
    nodes = list(map(int,raw_node_features[:, 0]))
    node_features = {n: list(features[i, :]) for i, n in enumerate(nodes)}

    # print(node_features)
    return G, node_neighbors, community, node_features

def load_data_matrix(net_file, com_file, feat_file):
    G = nx.read_edgelist(net_file,nodetype=int)
    # node_neighbors = {n: set(G.neighbors(n)) for n in G.nodes()}
    community = []
    with open(com_file, 'r') as f:
        for row in f:
            # nodes = row.strip().split()
            nodes = list(map(int, row.strip().split()))
            community.append(nodes)

    raw_node_features = np.genfromtxt(feat_file, dtype=np.dtype(str))
    features = np.array(raw_node_features[:, 1:], dtype=np.int32)
    nodes = list(map(int,raw_node_features[:, 0]))
    node_2_featId = {n: i for i, n in enumerate(nodes)}

    member_matrix = np.zeros((len(nodes),len(community)),dtype=np.float32)
    cid = 0
    for c in community:
        for n in c:
            id = node_2_featId[n]
            member_matrix[id][cid] = 1.
        cid += 1

    return G, features, member_matrix , community


if __name__ == '__main__':
    net_file = '../test/tiny.ugraph'
    com_file = '../test/tiny.cmty'
    feat_file = '../test/tiny.feat'

    G, node_neighbors, community, node_features = load_data(net_file, com_file, feat_file)
    #
    # density = cal_density(G, community, node_neighbors)
    # print('density:', density)
    #
    # node_sum = len(G.nodes())
    # entropy = cal_entropy(node_sum, community, node_features)
    # print('entropy:', entropy)
    #
    simJaccard = cal_simJaccard(community, node_features)
    print('simJaccard:', simJaccard)
    #
    # simCos = cal_simCos(community, node_features)
    # print('simCos:', simCos)
    #

    # a = cal_simED(community, node_features)

    # simED = cal_simED(community, node_features)
    # print('simED:', simED)
    #
    # EQ = modularity.cal_EQ(community, G)
    # print('EQ:', EQ)



    G, feats, member_matrix, community = load_data_matrix(net_file, com_file, feat_file)
    #
    # Q = cal_Q_M(G,member_matrix)
    # print("Q:",Q)

    # cal_Q(G,community)
    # print(feats)
    # print(member_matrix)

    cal_simCos_M(G, member_matrix,feats)

    # b = cal_simED_M(G, member_matrix,feats)

    # print(a  - b)

    cal_simJac_M(G, member_matrix,feats)

