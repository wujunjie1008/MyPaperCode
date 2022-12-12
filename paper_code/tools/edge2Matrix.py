import networkx as nx
def edge2Matrix(dataset):
    # path='./Datasets_Attributed Networks/'+dataset + '/network.txt'
    # path = './Datasets_Attributed Networks/' + dataset + '/remap/' + dataset + '.ugraph'
    # path = './Datasets_Attributed Networks/binary_networks(LFR)/' + dataset + '/network.txt'    # 人造网络路径
    path = './Datasets_Attributed Networks/1k_LFR/' + dataset + '/network.txt'  # 人造网络路径
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
    nodeNum = G.number_of_nodes()
    print(nodeNum)
    As = nx.adjacency_matrix(G)
    A = As.todense()
    return A

def outputFile(dataset):
    A = edge2Matrix(dataset).tolist()
    fileName = './Matrix/'+ dataset +'.txt'
    Note=open(fileName,mode='w')
    for i in range(len(A)):
        for j in range(len(A[i])):
            if(A[i][j]==1):
                A[j][i]=1

    for i in A:
        for j in range(len(i)):
            if i[j]==1:
                Note.write('1')
            else:
                Note.write('0')
            if j!= len(i)-1:
                Note.write(' ')
        Note.write('\n')
    Note.close()
# dataset = ['1k_0.1_hard','2k_0.1_hard','3k_0.1_hard','4k_0.1_hard','5k_0.1_hard']
# outputFile("texas")
dataset = ['3k_0.1mu_hard','3k_0.2mu_hard','3k_0.3mu_hard','3k_0.4mu_hard','3k_0.5mu_hard','3k_0.6mu_hard','3k_0.7mu_hard',
           '3k_0.8mu_hard','3k_0.9mu_hard']
for i in dataset:
    outputFile(i)