import geatpy as ea
import numpy as np
from scipy.spatial.distance import cdist
import StepTwo.ProblemQ_SSE as ProblemQ_SSE  # 导入自定义问题接口
import StepTwo.ProblemKKM_RC as ProblemKKM_RC
import multi_NSGA2.moea_multi_NSGA2 as multi_NSGA2
from tools.ARI import *

def test(dataset,maxIter,clusterNum):

    ds_name = dataset
    problem=[]
    """===============================实例化问题对象==========================="""
    problem.append(ProblemQ_SSE.myProblem(ds_name,clusterNum))  # 生成第一个问题对象
    problem.append(ProblemKKM_RC.myProblem(ds_name,clusterNum))  # 生成第二个问题对象
    """=================================种群设置=============================="""
    Encoding = 'P'  # 编码方式
    NIND = 100  # 种群规模
    PopNum = 2  # 种群数量
    Field = []
    for i in range(PopNum):
        Field.append(ea.crtfld(Encoding, problem[i].varTypes, problem[i].ranges, problem[i].borders)) # 创建区域描述器
    # 创建初始解
    # prophetChrom = findProphetCenters(ds_name, 10)
    # population = ea.Population(Encoding, Field, NIND, prophetChrom)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    population=[]  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    for i in range(PopNum):
        population.append(ea.Population(Encoding, Field[i], NIND))
    print(population)
    """===============================算法参数设置============================="""
    myAlgorithm = multi_NSGA2.moea_multi_NSGA2(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = maxIter  # 最大进化代数
    myAlgorithm.mutOper.Pm = 0.5
    myAlgorithm.logTras = 10  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置47是否打印输出日志信息
    myAlgorithm.drawing = 1  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化========================"""
    listRun = myAlgorithm.run()
    [BestIndi1, population1] = listRun[0]  # 执行算法模板，得到最优个体以及最后一代种群
    [BestIndi2, population2] = listRun[1]  # 执行算法模板，得到最优个体以及最后一代种群
    # print(population1)
    # 基因是vec_id的 . 读入时记得映射到node
    BestIndi1.save()  # 把最优个体的信息保存到文件中
    BestIndi2.save()  # 把最优个体的信息保存到文件中
    """=================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi1.sizes != 0:
        print('最优的模块度为：%s' % BestIndi1.ObjV[0][0])
        print('最优的sse为：%s' % BestIndi1.ObjV[0][1])
        print('最优的kkm为：%s' % BestIndi2.ObjV[0][0])
        print('最优的rc为：%s' % BestIndi2.ObjV[0][1])
        print('最优的控制变量值为：')
        print(BestIndi1.Phen[50])

        path = "./label/DataLabel/" + dataset + "Label.txt";
        fo1 = open(path, "r")
        trueLabel = [l.split() for l in fo1.readlines() if l.strip(" ")][0]
        for i in range(len(trueLabel)):
            trueLabel[i] = int(trueLabel[i])
        predictLabel = getResult(BestIndi1.Phen[50].tolist(), dataset, len(trueLabel))
        cm = clustering_metrics(trueLabel, predictLabel)
        cm.evaluationClusterModelFromLabel()

        # for i in range(BestIndi.Phen.shape[1]):
        # print(BestIndi.Phen[0, i])
    else:
        print('没找到可行解。')
    return BestIndi1.ObjV[0][0],BestIndi1.ObjV[0][1],BestIndi2.ObjV[0][0],BestIndi2.ObjV[0][1]

def getResult(centerNodeList,dataset,num):
    emb_path = './emb/open-ANE_emb/{:s}_node_embs.emb'.format(dataset)
    embDataTA = np.loadtxt(emb_path, dtype=float, delimiter=' ', skiprows=1)
    map_node2id = {}
    map_id2node = {}
    for i in range(embDataTA.shape[0]):
        node_id = int(embDataTA[i, 0])
        map_node2id[node_id] = i
        map_id2node[i] = node_id
    TAvectors = np.delete(embDataTA, 0, axis=1)  # 裁掉第一列的 nodeID
    TAF = TAvectors  # 计算向量间的相似性
    distTA = cdist(TAF, TAF, metric='sqeuclidean')
    cen_nodes_num = len(centerNodeList)
    distTA = distTA[:, centerNodeList]
    nearest = np.argmin(distTA, axis=1)  # 最近的中心的索引

    # 收集节点到社区
    comms = [[] for _ in range(cen_nodes_num)]
    comms_id = [[] for _ in range(cen_nodes_num)]

    for i in range(len(nearest)):
        comms[nearest[i]].append(map_id2node[i])
        comms_id[nearest[i]].append(i)
    print(comms)
    predLabel = [0] * num
    for i in range(len(comms)):
        for j in comms[i]:
            predLabel[j] = i
    return predLabel

list = [20]
res=[]
for i in list:
    q,sse,kkm,rc = test("cora",i,7)
    res.append([q,sse,kkm,rc])
    print(res)

# getResult([1763,1745,1382,938,1320,13,2679],"cora")