import numpy as np
import random
import matplotlib.pyplot as plt


def distance(point1, point2):  # 计算距离（欧几里得距离）
    return np.sqrt(np.sum((point1 - point2) ** 2))


def k_means(data, k, centerNode, max_iter=10000):
    centers = {}  # 初始聚类中心
    # 初始化，随机选k个样本作为初始聚类中心。 random.sample(): 随机不重复抽取k个值
    n_data = data.shape[0]  # 样本个数

    # for i in range(len(centerNode)):
    #     centers[i] = data[centers[i]]
    for idx, i in enumerate(random.sample(range(n_data), k)):
        # idx取值范围[0, k-1]，代表第几个聚类中心;  data[i]为随机选取的样本作为聚类中心
        centers[idx] = data[i]
    print(centers)
        # 开始迭代
    for i in range(max_iter):  # 迭代次数
        print("开始第{}次迭代".format(i + 1))
        clusters = {}  # 聚类结果，聚类中心的索引idx -> [样本集合]
        for j in range(k):  # 初始化为空列表
            clusters[j] = []

        for sample in data:  # 遍历每个样本
            distances = []  # 计算该样本到每个聚类中心的距离 (只会有k个元素)
            for c in centers:  # 遍历每个聚类中心
                # 添加该样本点到聚类中心的距离
                distances.append(distance(sample, centers[c]))
            idx = np.argmin(distances)  # 最小距离的索引
            clusters[idx].append(sample)  # 将该样本添加到第idx个聚类中心

        pre_centers = centers.copy()  # 记录之前的聚类中心点

        for c in clusters.keys():
            # 重新计算中心点（计算该聚类中心的所有样本的均值）
            centers[c] = np.mean(clusters[c], axis=0)

        is_convergent = True
        for c in centers:
            if distance(pre_centers[c], centers[c]) > 1e-8:  # 中心点是否变化
                is_convergent = False
                break
        if is_convergent == True:
            # 如果新旧聚类中心不变，则迭代停止
            break
    print(clusters)
    return centers, clusters


def predict(p_data, centers):  # 预测新样本点所在的类
    # 计算p_data 到每个聚类中心的距离，然后返回距离最小所在的聚类。
    distances = [distance(p_data, centers[c]) for c in centers]
    return np.argmin(distances)

def run(dataset):
    emb_path = './emb/open-ANE_emb/{:s}_node_embs.emb'.format(dataset)
    embDataTA = np.loadtxt(emb_path, dtype=float, delimiter=' ', skiprows=1)
    TAvectors = np.delete(embDataTA, 0, axis=1)  # 裁掉第一列的 nodeID
    centerNode = [2153, 2299, 72, 1382, 2558, 2006, 939]
    k_means(TAvectors, 7, centerNode, max_iter=10000)

run("cora")
