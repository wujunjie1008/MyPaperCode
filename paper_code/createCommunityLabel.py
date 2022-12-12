import networkx as nx
def create(dataset):    # 创建全是标签的形式
    if(dataset == "CiteSeer"):
        path = './Datasets_Attributed Networks/' + dataset + '/labels.txt'
        fo1 = open(path, "r")
        res = [l.split() for l in fo1.readlines() if l.strip(" ")]
        fileName = './label/DataLabel/' + dataset + 'Label.txt'
        Note = open(fileName, mode='w')
        Note.write(res[0][1])
        for i in res[1:]:
            Note.write(' ' + i[1])
        Note.close()
    else:
        # path = './Datasets_Attributed Networks/' + dataset + '/network.txt'
        path = './Datasets_Attributed Networks/1k_LFR/' + dataset + '/network.txt'  # 人造网络路径
        # path = './Datasets_Attributed Networks/' + dataset + '/remap/' + dataset + '.ugraph'
        G = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
        nodeNum = G.number_of_nodes()
        # path='./Datasets_Attributed Networks/'+dataset + '/communityLabel.txt'
        # path = './Datasets_Attributed Networks/' + dataset + '/remap/' + dataset + '.cmty'
        # path = './Datasets_Attributed Networks/binary_networks(LFR)/' + dataset + '/communityLabel.txt'  # 人造网络路径
        path = './Datasets_Attributed Networks/1k_LFR/' + dataset + '/community.txt'  # 人造网络路径
        fo1 = open(path, "r")
        lines2 = [l.split() for l in fo1.readlines() if l.strip(" ")]
        res = nodeNum*['']
        length = 0
        for i in lines2:
            length += len(i)
            for j in i:
                res[int(j)]=str(lines2.index(i))
        print(length)
        print(len(res))
        fileName = './label/DataLabel/' + dataset + 'Label.txt'
        Note = open(fileName, mode='w')
        Note.write(res[0])
        for i in res[1:]:
            Note.write(' '+i)
        Note.close()

def createForm2(dataset):    # 创建一个点对应一个标签的形式
    path = './label/DataLabel/' + dataset + 'Label.txt'
    fo1 = open(path, "r")
    res = [l.split() for l in fo1.readlines() if l.strip(" ")]
    path = './Datasets_Attributed Networks/' + dataset + '/communityLabel.txt'
    Note = open(path, mode='w')
    Note.write('0 '+res[0][0])
    for i in range(1,len(res[0])):
        Note.write('\n' + str(i) + ' ' + res[0][i])
    Note.close()
# dataset = ['1k_0.1_hard', '2k_0.1_hard', '3k_0.1_hard', '4k_0.1_hard', '5k_0.1_hard']
dataset = ['3k_0.1mu_hard','3k_0.2mu_hard','3k_0.3mu_hard','3k_0.4mu_hard','3k_0.5mu_hard','3k_0.6mu_hard','3k_0.7mu_hard',
           '3k_0.8mu_hard','3k_0.9mu_hard']
# dataset = ['texas','cornell','washington','wisconsin']
for i in dataset:
    create(i)
