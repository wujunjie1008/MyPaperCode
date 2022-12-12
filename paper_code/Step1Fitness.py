import networkx as nx
import math

def Sep(c, U):  # 分散性指标
    min = S(U[0], U[1])
    for i in range(c):
        j = i+1
        while(j < c):
            temp = S(U[i], U[j])
            min = temp if temp < min else min
            j += 1
    return 1-min


def S(UA,UB):
    max = UA[0] if UA[0] < UB[0] else UB[0]
    for i in range(len(UA)):
        # print("UA[i]=",UA[i])
        # print("UB[i]=", UB[i])
        temp = UA[i] if UA[i]<UB[i] else UB[i]
        # print("temp=", temp)
        max = temp if temp > max else max
    # print("max=", max)
    return max


def Lam(entropyList, DintList, DextList):  # 紧凑性指标 entropy/(Dint-Dext)
    print("entropyList:",entropyList)
    print("DintList:", DintList)
    print("DextList:", DextList)
    res = 0
    for i in range(len(entropyList)):
        res += entropyList[i]/(DintList[i]-DextList[i])
    return res


def entropy(attributeMatrix, dimension, community):  # 紧凑性指标中的属性部分
    res = []
    for cluster in community:
        result = 0
        for a in range(dimension):
            x = 0
            for n in cluster:
                if attributeMatrix[n][a] == 1:
                    x += 1
            x = x/len(cluster)
            if x != 0 and x != 1:
                # result += -1 * (x * math.log(x, 2) + (1-x) * math.log(1-x, 2))  # 网上信息熵以2为底数
                result += -1 * (x * math.log(x, math.e) + (1 - x) * math.log(1 - x, math.e))    # 论文中信息熵以e为底数
        res.append(result)

    return res

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
    print(entropy)
    return entropy

def Dint(community, G):  # 紧凑性指标中的结构部分：内密度
    res = []
    for communityIndex in range(len(community)):
        # 计算如果是完全图，簇内节点的边（分母）
        cluster_node_num = len(community[communityIndex])
        Ei = cluster_node_num * (cluster_node_num-1) / 2
        Einner = 0
        # 计算簇内节点的边（分子）
        for i in community[communityIndex]:
            for j in nx.neighbors(G, i):
                if j in community[communityIndex] and j>i:
                    Einner += 1
        if Ei == 0:
            Ei = 1
        res.append(Einner/Ei)
    # print("Dint:", res)
    return res


def Dext(community, G):  # 紧凑性指标中的结构部分：间密度
    res = []
    for communityIndex in range(len(community)):
        # 计算如果是完全图，簇外节点的边（分母）
        cluster_node_num = len(community[communityIndex])
        Eo = (len(G.nodes()) - cluster_node_num) * cluster_node_num
        Eout = 0
        # 计算簇外节点的边（分子）
        for i in community[communityIndex]:
            for j in nx.neighbors(G, i):
                if j not in community[communityIndex]:
                    Eout += 1
        res.append(Eout / Eo)
    # print("Dext:",res)
    return res

def cal_KKM_RC(partitions,G):

    comm_num = len(partitions)
    intra_score = 0
    inter_score = 0
    n = G.number_of_nodes()
    for partition in partitions:
        partition = set(partition)
        intra = 0
        inter = 0
        for node in partition:
            # if node not in G:
                # continue
            for nei in G.neighbors(node):
                if nei in partition:
                    intra += 1
                else:
                    inter += 1
        intra_score += intra / len(partition)
        inter_score += inter / len(partition)

    KKM = 2*(n - comm_num) - intra_score
    RC = inter_score
    # print(comm_num,n)
    return KKM, RC

