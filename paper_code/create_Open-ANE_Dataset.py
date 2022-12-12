def createLabel(dataset):
    # fileName = './label/DataLabel/' + dataset + 'Label.txt'
    fileName = './Datasets_Attributed Networks/' + dataset + '/labels.txt'
    # fileName = './Datasets_Attributed Networks/binary_networks(LFR)/' + dataset + '/network.txt'  # 人造网络路径
    fo1 = open(fileName, "r")
    res = [l.split() for l in fo1.readlines() if l.strip(" ")]
    fileName = './open-ANE_Dataset/'+dataset+'/' + dataset + '_label.txt'
    Note = open(fileName, mode='w')
    Note.write('0 '+res[0][0])
    for i in range(1,len(res[0])):
        Note.write('\n' + str(i) + ' ' + res[0][i])
    Note.close()
def createAdjList(dataset):
    fileName = './Datasets_Attributed Networks/' + dataset + '/network.txt'
    # fileName = './Datasets_Attributed Networks/' + dataset + '/remap/' + dataset + '.ugraph'
    # fileName = './Datasets_Attributed Networks/binary_networks(LFR)/' + dataset + '/network.txt'  # 人造网络路径
    fo1 = open(fileName, mode='r')
    res = [l.split() for l in fo1.readlines() if l.strip(" ")]
    pre=''
    fileName = './open-ANE_Dataset/' + dataset + '/' + dataset + '_adjlist.txt'
    Note = open(fileName, mode='w')
    for i in range(len(res)):
        if(res[i][0]!=pre):
            pre = res[i][0]
            Note.write('\n')
            Note.write(res[i][0]+' ')
            Note.write(res[i][1])
            continue
        else:
            Note.write(' ' + res[i][1])

    Note.close()

def run(dataset):
    createLabel(dataset)
    createAdjList(dataset)
# dataset = ['1k_0.1_hard','2k_0.1_hard','3k_0.1_hard','4k_0.1_hard','5k_0.1_hard']
# dataset = ['texas','CiteSeer','cora','cornell','washington','wisconsin']
dataset = ['wiki']
for i in dataset:
    run(i)