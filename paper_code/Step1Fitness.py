import networkx as nx
import math

def Sep():  # 分散性指标
    return None


def Lam(entropyList,DintList,DextList):  # 紧凑性指标 entropy/(Dint-Dext)
    res = []
    for i in range(len(entropyList)):
        res.append(entropyList[i]/(DintList[i]-DextList[i]))
    return res


def entropy(attributeMatrix,community,G):  # 紧凑性指标中的属性部分
    res = []
    result = 0
    for cluster in community:
        result = 0
        for a in range(len(attributeMatrix)):
            x = 0
            for n in cluster:
                if attributeMatrix[n][a] == 1:
                    x += 1
            x = x/len(cluster)
            result += -1 * (x * math.log(x, 2) + (1-x) * math.log(1-x, 2))  # 网上信息熵以2为底数
            # result += -1 * (x * math.log(x, math.e) + (1 - x) * math.log(1 - x, math.e))    # 论文中信息熵以e为底数
            res.append(result)
    return res


def Dint(community,G):  # 紧凑性指标中的结构部分：内密度
    res = []
    for communityIndex in range(len(community)):
        # 计算如果是完全图，簇内节点的边（分母）
        Ei = 0
        for i in range(1, len(community[communityIndex])):
            Ei = Ei + i
        Einner = 0
        # 计算簇内节点的边（分子）
        for i in community[communityIndex]:
            for j in nx.neighbors(G, i):
                if j in community[communityIndex] and j>i:
                    Einner += 1
        res.append(Einner/Ei)
    return res


def Dext(community,G):  # 紧凑性指标中的结构部分：间密度
    res = []
    for communityIndex in range(len(community)):

        # 计算如果是完全图，簇外节点的边（分母）
        Ei = 0
        for i in range(1, len(community[communityIndex])):
            Ei = Ei + i
        Eo = len(G.edge())-Ei

        Eout = 0
        # 计算簇外节点的边（分子）
        for i in community[communityIndex]:
            for j in nx.neighbors(G, i):
                if j not in community[communityIndex]:
                    Eout += 1
        res.append(Eout / Eo)
        print(Eo)
        print(Eout)
    return res

a = [[0,1,2,3,4]]
b = [[3],[2,4,5,6],[1,3,4],[0,2,7],[1,2]]
Dint(a,b)

