from sklearn.cluster import KMeans
from sklearn import cluster
import numpy as np
def run(agri,dataset,num):
    # 生成10*3的矩阵
    emb_path = './emb/'+agri+'_emb/{:s}_node_embs.txt'.format(dataset)
    embDataTA = np.loadtxt(emb_path, dtype=float, delimiter=' ', skiprows=1)
    # 聚类为4类
    estimator = KMeans(n_clusters=num)
    # fit_predict表示拟合+预测，也可以分开写
    res = estimator.fit_predict(embDataTA)
    # 预测类别标签结果
    lable_pred = estimator.labels_
    # 各个类别的聚类中心值
    centroids = estimator.cluster_centers_
    # 聚类中心均值向量的总和
    inertia = estimator.inertia_
    res = lable_pred.tolist()
    path = "./label/" + agri + "Res/" + dataset + ".txt";
    Note = open(path, mode='w')
    Note.write(str(res[0]))
    for i in res[1:]:
        Note.write(' ' + str(i))
    Note.close()
    return res
# dataset = ['texas','CiteSeer','cora','cornell','washington','wisconsin']
# num = [5,6,7,5,5,5]

dataset = ['1k_0.1_hard','2k_0.1_hard','3k_0.1_hard','4k_0.1_hard','5k_0.1_hard']
num = [20,51,84,104,135]
for i in range(len(dataset)):
    # run("deepwalk",dataset[i],num[i])
    run("node2vec", dataset[i], num[i])