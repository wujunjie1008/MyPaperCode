from tools.ARI import *
from  StepTwo import KMeans
def cal(agri,dataSet):
    print("===============",dataSet,"===============")
    path="./label/"+agri+"Res/"+dataSet+".txt";
    fo1 = open(path, "r")
    predictLabel = [l.split() for l in fo1.readlines() if l.strip(" ")][0]
    for i in range(len(predictLabel)):
        predictLabel[i] = int(predictLabel[i])
    path = "./label/DataLabel/" + dataSet + "Label.txt";
    fo1 = open(path, "r")
    trueLabel = [l.split() for l in fo1.readlines() if l.strip(" ")][0]
    for i in range(len(trueLabel)):
        trueLabel[i] = int(trueLabel[i])
    cm = clustering_metrics(trueLabel,predictLabel)
    cm.evaluationClusterModelFromLabel()
# dataset = ['washington','CiteSeer','texas','cora','cornell','wisconsin']
# # dataset = ['1k_0.1_hard','2k_0.1_hard','3k_0.1_hard','4k_0.1_hard','5k_0.1_hard']
# for i in dataset:
#     # cal("node2vec", i)
#     # cal("RMOEA", i)
#     # cal("SCI", i)
#     cal("Hang", i)
