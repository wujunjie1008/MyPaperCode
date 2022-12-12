import networkx as nx
import numpy as np
import math
import Step2Fitness.modularity as modularity
from scipy.spatial.distance import cdist


def cal_sim_M_B(mem_matrix,B):
    k = mem_matrix.shape[1]
    H = mem_matrix

    memSizeArray = np.array(H.T.sum(1))
    memSizeDiag = np.diag(memSizeArray * (memSizeArray - 1) - 1)
    ones = np.ones_like(memSizeDiag)
    norm_matrix = ones + memSizeDiag

    norm_M = np.divide(H.T.dot(B).dot(H), norm_matrix, out=np.zeros_like(norm_matrix), where=norm_matrix != 0)
    sim = np.trace(norm_M) / k
    return sim

# 根据特征举证计算相似度矩阵
def gen_SimMatrix(feat_matrix, metrics='cosine', sigma=5):
    # 对角元素是 0.. 要特殊处理
    X = feat_matrix
    if metrics == 'cosine':
        A_inner = X.dot(X.T)  # 相似度矩阵
        A_inner = A_inner - np.diag(A_inner.diagonal())  # 去对角元素
        A_norm = np.linalg.norm(X, axis=1,keepdims=True)  # 计算每个向量的范数
        A_ij_norm = A_norm.dot(A_norm.T)  # 两两范数相乘
        B = np.divide(A_inner, A_ij_norm, out=np.zeros_like(A_ij_norm), where=A_ij_norm != 0)

        # dist = cdist(X, X, metric='cosine')
        # B = 1 - dist
        # B[np.isnan(B)] = 0
        # B = B - np.diag(B.diagonal())  # 去对角元素

    elif metrics == 'hamming':
        # hamming 距离统计两向量不同的计数个数
        # 将原数据+1,并内积 可得到三种属性组合: {1 * 1} {1 * 2} {2 * 2} 需要统计的是 {1 * 2} 的个数
        # {1*1} + 2 {1*2} + 4 {2*2} = 新矩阵
        # {1 * 1}的出现次数 + 1/2 {1 * 2}的出现次数 + 1/4 {2 * 2}的出现次数 = 属性列数
        # {2 * 2}的出现次数为原矩阵相乘(即原矩阵中同为1的交集) 得到如下代码
        X_plus = X + 1
        X_sum = X_plus.dot(X_plus.T)
        Xint_1 = X.dot(X.T)
        X_sum = X_sum - X.shape[1] - 3 * Xint_1
        dist = X_sum / X.shape[1]
        B = 1 - dist
        B = B - np.diag(B.diagonal())  # 去对角元素

    elif metrics == 'jaccard':
        # 假设输入的只有 0,1
        # jaccard = 交集/并集, 交集简单, 只要矩阵相乘即可
        # 并集需要矩阵相乘的计算方式, 但元素之间用异或而不是乘, 但是好像没有这样的函数
        # 由于输入只有 0,1 对每个元素加上1 然后矩阵相乘. 加1后向量之间的相乘 分三种情况 {1 * 1} {1 * 2}(或 {2 * 1}) {2 * 2}
        # 我们需要计算并集，实际上就是 {2 * 2} 和 {1 * 2} 出现的次数. 此外需要减去一部分值, 与下面有关:
        # 1. {1 * 1}的出现次数, 即原本都为0
        # 2. {1 * 2}出现次数, 即原来一个是1, 一个是0. 由于1*2 = 2 我们要减去多计算的一次
        # 3. {2 * 2}的出现次数, 即原来都是1. 由于2*2 = 4 要多减去三次
        # 综合(1) (2) (3) 减去属性总数后，还需要再减去 {2 * 2}还多出来的两次
        # {2 * 2} 实际上就是交集, 也就是再减去两次交集.

        X_plus = X + 1
        X_sum = X_plus.dot(X_plus.T)
        X_sum = X_sum - X_plus.shape[1]
        Xint_1 = X.dot(X.T)
        X_sum = X_sum - Xint_1 * 2

        dist = np.divide(Xint_1, X_sum, out=np.zeros_like(X_sum), where=X_sum != 0)
        B = dist - np.diag(dist.diagonal())  # 去对角元素

        # print(X_sum[0][0],X_sum[0][1],X_sum[0][2])
        # print(Xij_sum.shape,Xjac.shape)
        # dist = cdist(X, X, metric='jaccard')
        # dist = 1 - dist
    elif metrics == 'euclidean':
        X_ij = X.dot(X.T)
        X_sq = np.sum(X * X, axis=1, keepdims=True)
        Y_sq = X_sq.T
        dist = -2 * X_ij + X_sq + Y_sq

        # X_sq 和 Y_sq 互为转置作用到dist矩阵上
        # dist = cdist(X,X,metric = 'euclidean')
        # dist = cdist(X, X, metric='sqeuclidean')  # square of euclidean
        # B = 1 / (1 + dist)
        # B = dist
        # K(i, j) = exp(- | x_i - x_j | ^ 2 / (2 * sigma) ^ 2)
        B = np.exp(- dist / (2 * sigma * sigma))
        B = B - np.diag(B.diagonal())  # 去对角元素
    else:
        raise Exception('unknown similarity meaturement')

    return B

def cal_sim_M(mem_matrix, feat_matrix, metrics='cosine', sigma=5) :
    X = feat_matrix
    if metrics == 'cosine':
        A_inner = X.dot(X.T)  # 相似度矩阵
        A_inner = A_inner - np.diag(A_inner.diagonal())  # 去对角元素
        A_norm = np.linalg.norm(X, axis=1,keepdims=True)  # 计算每个向量的范数
        # A_norm = A_norm.reshape((A_norm.shape[0], -1))
        A_ij_norm = A_norm.dot(A_norm.T)  # 两两范数相乘
        B = np.divide(A_inner, A_ij_norm, out=np.zeros_like(A_ij_norm), where=A_ij_norm != 0)

        # dist = cdist(X, X, metric='cosine')
        # B = 1 - dist
        # B[np.isnan(B)] = 0
        # B = B - np.diag(B.diagonal())  # 去对角元素

    elif metrics == 'hamming':
        # hamming 距离统计两向量不同的计数个数
        # 将原数据+1,并内积 可得到三种属性组合: {1 * 1} {1 * 2} {2 * 2} 需要统计的是 {1 * 2} 的个数
        # {1*1} + 2 {1*2} + 4 {2*2} = 新矩阵
        # {1 * 1}的出现次数 + 1/2 {1 * 2}的出现次数 + 1/4 {2 * 2}的出现次数 = 属性列数
        # {2 * 2}的出现次数为原矩阵相乘(即原矩阵中同为1的交集) 得到如下代码
        X_plus = X + 1
        X_sum = X_plus.dot(X_plus.T)
        Xint_1 = X.dot(X.T)
        X_sum = X_sum - X.shape[1] - 3 * Xint_1
        dist = X_sum / X.shape[1]
        B = 1 - dist
        B = B - np.diag(B.diagonal())  # 去对角元素

    elif metrics == 'jaccard':
        # 假设输入的只有 0,1
        # jaccard = 交集/并集, 交集简单, 只要矩阵相乘即可
        # 并集需要矩阵相乘的计算方式, 但元素之间用异或而不是乘, 但是好像没有这样的函数
        # 由于输入只有 0,1 对每个元素加上1 然后矩阵相乘. 加1后向量之间的相乘 分三种情况 {1 * 1} {1 * 2}(或 {2 * 1}) {2 * 2}
        # 我们需要计算并集，实际上就是 {2 * 2} 和 {1 * 2} 出现的次数. 此外需要减去一部分值, 与下面有关:
        # 1. {1 * 1}的出现次数, 即原本都为0
        # 2. {1 * 2}出现次数, 即原来一个是1, 一个是0. 由于1*2 = 2 我们要减去多计算的一次
        # 3. {2 * 2}的出现次数, 即原来都是1. 由于2*2 = 4 要多减去三次
        # 综合(1) (2) (3) 减去属性总数后，还需要再减去 {2 * 2}还多出来的两次
        # {2 * 2} 实际上就是交集, 也就是再减去两次交集.

        X_plus = X + 1
        X_sum = X_plus.dot(X_plus.T)
        X_sum = X_sum - X_plus.shape[1]
        Xint_1 = X.dot(X.T)
        X_sum = X_sum - Xint_1 * 2

        dist = np.divide(Xint_1, X_sum, out=np.zeros_like(X_sum), where=X_sum != 0)
        B = dist - np.diag(dist.diagonal())  # 去对角元素

        # print(X_sum[0][0],X_sum[0][1],X_sum[0][2])
        # print(Xij_sum.shape,Xjac.shape)
        # dist = cdist(X, X, metric='jaccard')
        # dist = 1 - dist
    elif metrics == 'euclidean':
        X_ij = X.dot(X.T)
        X_sq = np.sum(X * X, axis=1, keepdims=True)
        Y_sq = X_sq.T
        dist = -2 * X_ij + X_sq + Y_sq

        # X_sq 和 Y_sq 互为转置作用到dist矩阵上
        # dist = cdist(X,X,metric = 'euclidean')
        # dist = cdist(X, X, metric='sqeuclidean')  # square of euclidean
        # B = 1 / (1 + dist)
        # B = dist
        # K(i, j) = exp(- | x_i - x_j | ^ 2 / (2 * sigma) ^ 2)
        B = np.exp(- dist / (2 * sigma * sigma))
        B = B - np.diag(B.diagonal())  # 去对角元素
    else:
        raise Exception('unknown similarity meaturement')

    k = mem_matrix.shape[1]
    H = mem_matrix

    memSizeArray = np.array(H.T.sum(1))
    memSizeDiag = np.diag(memSizeArray * (memSizeArray - 1) - 1)
    ones = np.ones_like(memSizeDiag)
    norm_matrix = ones + memSizeDiag

    norm_M = np.divide(H.T.dot(B).dot(H), norm_matrix, out=np.zeros_like(norm_matrix), where=norm_matrix != 0)

    sim = np.trace(norm_M) / k

    return sim

def cal_simCos_M(mem_matrix, feat_matrix):
    """
    calculate cosine similarity using member matrix
    :param community:
    :param node_features:
    :return:
    """
    A_inner = feat_matrix.dot(feat_matrix.T) # 相似度矩阵
    A_inner = A_inner - np.diag(A_inner.diagonal()) # 去对角元素
    A_norm = np.linalg.norm(feat_matrix,axis=1) # 计算每个向量的范数
    A_norm = A_norm.reshape((A_norm.shape[0],-1))
    A_ij_norm = A_norm.dot(A_norm.T) # 两两范数相乘
    #
    k = mem_matrix.shape[1]
    # B = A_inner / A_ij_norm # 得到变换矩阵
    B = np.divide(A_inner, A_ij_norm, out=np.zeros_like(A_ij_norm), where=A_ij_norm != 0)
    # B[np.isnan(B)] = 0
    # B = B - np.diag(B.diagonal())  # 去对角元素

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

def cal_simHamming_M(mem_matrix, feat_matrix):
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
    # print(1)
    dist = cdist(X,X,metric = 'jaccard')
    # print(2)
    dist = 1 - dist
    B = dist - np.diag(dist.diagonal())  # 去对角元素

    k = mem_matrix.shape[1]
    H = mem_matrix

    memSizeArray = np.array(mem_matrix.T.sum(1))
    memSizeDiag = np.diag(memSizeArray * (memSizeArray - 1) - 1)
    ones = np.ones_like(memSizeDiag)
    norm_matrix = ones + memSizeDiag

    norm_M = np.divide(H.T.dot(B).dot(H), norm_matrix, out=np.zeros_like(norm_matrix), where=norm_matrix != 0)
    simJac = np.trace(norm_M)/ k

    return simJac

def gen_Q_B(G,F,sub=1):
    """
    generate modularity matrix B for calculating modularity Q
    equation: B = B = A - Dij / (2 * m)
    where m is the number of edges in graph, A is the adjacency matrix and D is the degree matrix.
    :return: modularity matrix B
    """
    m = G.number_of_edges()
    # n = G.number_of_nodes()
    n = F.shape[0]
    A = np.zeros((n,n))
    # A = np.array(nx.adjacency_matrix(G).todense())
    for u,v in G.edges():
        u -= sub
        v -= sub
        A[u][v] = 1
        A[v][u] = 1

    D = np.array(A.sum(1)).reshape(-1,1)
    Dij = D.dot(D.T)
    B = A - Dij / (2 * m)

    return B

def cal_Q_M_B(M,B,m):
    H = M
    Q = np.trace(H.T.dot(B).dot(H))/ (2 * m)
    return Q

def cal_Q_M(G,mem_matrix,sub=1):
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
        u -= sub
        v -= sub
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

def cal_sim(community, S):
    k = len(community)
    simCos = 0
    attr_links = 0
    node_num = 0
    for c in community:
        if c is None:
            k -=1
            continue
        if len(c) == 1:
            continue

        comSim = 0
        for i in range(len(c) - 1):
            node_num += 1
            for j in range(i + 1, len(c)):
                comSim += S[c[i],c[j]]

        # simCos += len(c) * 2 * comSim / (len(c) * (len (c) - 1))
        # simCos +=  2 * comSim / (len(c) * (len (c) - 1))
        simCos += comSim
        attr_links += (len(c) * (len (c) - 1)) / 2

    # return simCos / node_num
    # return simCos / k
    return simCos / attr_links

# 给定一个社区划分, 以及节点属性
# 计算余弦相似度指标.
# community = [[0,2,3],[4,6,7],[],...]
def cal_simCos(community, node_features):
    # i < j
    k = len(community)
    simCos = 0
    attr_links = 0
    node_num = 0
    for c in community:
        if c is None:
            k -=1
            continue
        if len(c) == 1:
            continue

        comSim = 0
        node_num += len(c)
        for i in range(len(c) - 1):
            for j in range(i + 1, len(c)):
                a = np.array(node_features[c[i]])
                b = np.array(node_features[c[j]])
                nume = np.inner(a,b)
                denomi = np.linalg.norm(a) * np.linalg.norm(b)
                cos_sim = nume / denomi
                comSim += cos_sim

        # simCos += len(c) * 2 * comSim / (len(c) * (len (c) - 1))
        # simCos +=  2 * comSim / (len(c) * (len (c) - 1))
        simCos += comSim
        attr_links += (len(c) * (len (c) - 1)) / 2
    # print(node_num)
    # return simCos / node_num
    return simCos / attr_links
    # return simCos / k

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


def cal_Q(G, community):
    m = G.number_of_edges()
    Qsum = 0
    for c in community:
        sum = 0
        for i in c:
            for j in c:
                if G.has_edge(i,j):
                    sum += 1 - G.degree(i) * G.degree(j) / (2*m)
                    # sum += 1
                else:
                    sum += 0 - G.degree(i) * G.degree(j) / (2*m)
                    # sum += 0
        # print(c, sum)
        Qsum += sum
    # print(Qsum / (2 * m))
    return Qsum / (2 * m)


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
    features = np.array(raw_node_features[:, 1:], dtype=np.float64)
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
    features = np.array(raw_node_features[:, 1:], dtype=np.float64)
    nodes = list(map(int,raw_node_features[:, 0]))
    node_2_featId = {n: i for i, n in enumerate(nodes)}

    member_matrix = np.zeros((len(nodes),len(community)),dtype=np.float64)
    cid = 0
    for c in community:
        for n in c:
            id = node_2_featId[n]
            member_matrix[id][cid] = 1.
        cid += 1

    return G, features, member_matrix , community


def euclidean_distances(x, y, squared=True):
    # ref: https://www.cnblogs.com/quarryman/p/euclidean_distances.html (calc euclidean similarity by matrix)
    # ref: https://zhuanlan.zhihu.com/p/27739282 (what is np.einsum)

    """Compute pairwise (squared) Euclidean distances.
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]

    x_square = np.sum(x * x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y * y, axis=1, keepdims=True).T

    distances = np.dot(x, y.T)

    # print(y_square.shape,x_square.shape,distances.shape)
    # use inplace operation to accelerate
    distances *= -2
    print(y_square)
    # print(distances)
    distances += x_square
    # print(distances)
    distances += y_square
    # print(distances)

    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)

    print(distances[0][1],distances[0][2])
    if x is y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances


if __name__ == '__main__':
    net_file = '../test/tiny.ugraph'
    com_file = '../test/tiny.cmty'
    feat_file = '../test/tiny.feat'

    from tools.Loader import Loader
    from tools import file_io
    loader = Loader(net_file,feat_file)
    G,F = loader.get_GFMC()

    com = file_io.read_communities(com_file,False)
    a = cal_simCos(com,F)
    print(a)

