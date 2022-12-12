import numpy as np
import networkx as nx
import math
from gensim.models import Word2Vec
from sklearn import preprocessing
from collections import defaultdict
import time
from gensim import matutils
from annoy import AnnoyIndex
import random

def load_graph(path):
    G = nx.Graph()
    with open(path) as text:
        for line in text:
            vertices = line.strip().split()
            # print(vertices[0],vertices[1])
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G

# generate a sequence of nodes by random walk from start node
def random_walk(G, start, walk_length):
    walk = [start]
    for i in range(walk_length-1):
        cur = walk[-1]
        next = np.random.choice(list(G[int(cur)]))
        walk.append(next)
    return walk
    # return [str(x) for x in  walk]

def build_corpus(G, stru_num_dict, walk_length):
    walks = []
    for node in G.nodes():
        for i in range(stru_num_dict[node]):
            if len(G[node]) == 0:
                walks.append([str(node)])
                continue
            walk = random_walk(G, node, walk_length)
            walks.append(walk)
    return walks

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def r_neighborhood(G,start,r):
    r_neibors = dict() # key 节点, value 是start的几阶节点
    temp = [start] # temp存放当前待拓展邻居的节点，初始化是源节点
    neibors_last = set([start]) # neibors_last 存放之前已经记录过最短路径的节点，保证过去已经有的节点不重复扩展
    neibors_add = set() # 存放当前轮新增节点
    for i in range(r):
        neibors = set() # 存放新一轮扩展的所有节点，可能包含过去已经扩展过的节点
        for e in temp:
            neibors = neibors.union(set(G[e]))
        neibors_add = neibors - neibors_last
        temp = neibors_add
        for e in neibors_add:
            if e not in r_neibors:
                r_neibors[e] = i+1
                neibors_last.add(e)
    return r_neibors

def cal_p_r_neibors(G,r):
    nodes_pij_dic = defaultdict(dict)
    nodes_r_neibors = dict()
    for node in G.nodes():
        r_neibors = r_neighborhood(G,node,r)
        nodes_r_neibors[node] = r_neibors
        # print(node, len(r_neibors))
        p_dic = dict()
        for i,d in r_neibors.items():
            if d == 1:
                p_dic[i] = 1.0
            elif d == 2:
                pub_nei = set(G[node]) & set(G[i])
                num_pub_nei = len(pub_nei)
                # p_dic[i] = num_pub_nei/(np.sqrt(G.degree(node)*G.degree(i)))
                p_dic[i] = num_pub_nei / (G.degree(node) + G.degree(i)-num_pub_nei)
        nodes_pij_dic[node] = p_dic
        # print(p_dic)
    return nodes_pij_dic

def cal_node_entropy(G):
    nodes_entropy = defaultdict(float)
    total_degree = 2 * G.number_of_edges()
    # print("total_degree: ", total_degree)
    for node in G.nodes():
        e = 0.0
        for nei in G[node]:
            p_nei = G.degree(nei) / total_degree
            e += - p_nei * math.log2(p_nei)
        nodes_entropy[node] = e
        # print(nodes_entropy[node],len(G[node]))
    return nodes_entropy

def cal_window_entropy(center, windows, nodes_pij_dic, nodes_entropy):
    window_entropy = 0
    for window in windows:
        for i in window:
            if i in nodes_pij_dic[center]:
                window_entropy += (1-nodes_pij_dic[center][i]) * (nodes_entropy[center]+nodes_entropy[i])
            else:
                window_entropy += (nodes_entropy[center] + nodes_entropy[i])
    # print(window_entropy)
    return window_entropy

def cal_sequence_entropy(sequences, window_size, nodes_pij_dic, nodes_entropy):
    seq_and_seqEntropy_list = []
    for seq in sequences:
        walk_length = len(seq)
        seq_entropy = 0
        for index, center in enumerate(seq):
            windows = []
            l = index-window_size
            r = index+window_size
            if l < 0:
                left_window = seq[0:index]
            else:
                left_window = seq[l:index]
            if r > (walk_length-1):
                right_window = seq[index+1:]
            else:
                right_window = seq[index+1:r+1]
            windows.append(left_window)
            windows.append(right_window)
            window_entropy = cal_window_entropy(center, windows, nodes_pij_dic, nodes_entropy)
            seq_entropy += window_entropy
        # print(seq_entropy)
        seq_and_seqEntropy = (seq_entropy, seq)
        # print(seq_and_seq_entropy)
        seq_and_seqEntropy_list.append(seq_and_seqEntropy)
    seq_and_seqEntropy_list = sorted(seq_and_seqEntropy_list,reverse=True)
    return seq_and_seqEntropy_list

def refine_sequences(seq_and_seqEntropy_list, ratio):
    refine_seqs = []
    num_seq = len(seq_and_seqEntropy_list)
    num_preserve = int((1-ratio) * num_seq)
    for i in range(num_preserve):
        refine_seqs.append([str(x) for x in seq_and_seqEntropy_list[i][1]])
    return refine_seqs

# 默认属性值是浮点数
def load_attributes(file):
    attri_dict = dict()
    flag = True
    with open(file) as text:
        for line in text:
            lis = line.strip().split()
            attri_dict[int(lis[0])] = [float(i) for i in lis[1:]]
            if flag:
                num_of_features = len(attri_dict[int(lis[0])])
                flag = False
    return attri_dict, num_of_features
# Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot".
def build_annoyTree(attri_dict, num_of_features):
    tree = AnnoyIndex(num_of_features, 'dot')
    for k in attri_dict.keys():
        tree.add_item(k, attri_dict[k])
    tree.build(50)
    return tree

def build_attribute_corpus(G, attr_num_dict, annoy_tree, k, attri_dict, walk_length):
    walks_attr = []
    k_neibors = dict()
    F = np.zeros((G.number_of_nodes(), G.number_of_nodes()))

    for node in G.nodes():
        k_neibors[node] = annoy_tree.get_nns_by_item(node, k)
        for i in k_neibors[node]:
            F[node][i] = annoy_tree.get_distance(node,i)/(np.sqrt(sum(attri_dict[node]))*np.sqrt(sum(attri_dict[i])))

    for node in G.nodes():
        for j in range(attr_num_dict[node]):
            m = random.randint(2, k)
            walk_attr = k_neibors[node][:m]
            walk_attr.append(node)
            random.shuffle(walk_attr)
            walks_attr.append([str(s) for s in walk_attr])
    return walks_attr, F

def row_norm(A):
    for row in range(A.shape[0]):
        sm = sum(A[row])
        if sm != 0:
            A[row] = A[row]/sm


def cal_attr_num_walk(attri_dict, attr_numwalk):
    total = len(attri_dict) * attr_numwalk
    attr_num_dict = dict()
    div = 0
    for node, vs in attri_dict.items():
        div += sum(vs)
    for node, attrs in attri_dict.items():
        attr_num_dict[node] = int(sum(attrs) * total/div)
    return attr_num_dict

def cal_stru_num_walk(G, stru_numwalk):
    total = G.number_of_nodes() * stru_numwalk
    stru_num_dict = dict()
    div = 2 * G.number_of_edges()
    for node in G.nodes():
        stru_num_dict[node] = int(len(G[node]) * total/div)
    return stru_num_dict

def main(para_dict):
    start_t = time.time()
    # ****************************************************************************************
    # 算法参数
    # 输入输出参数
    graph = para_dict['graph']
    input_graph = graph + '/network.txt'
    input_attribute = graph + '/features.txt'
    emb_file = graph + '/' + para_dict['emb_file']
    enhance_emb_file = graph + '/' + para_dict['enhance_emb_file']
    # 实验中固定的算法参数
    representation_size = 128
    walk_length = 10
    window_size = 10
    total_numWalks = 80
    T = 3
    lambda1 = 0.5
    lambda2 = 0.25
    workers = 4
    # 实验中调整的算法参数
    ratio_structure_numWalks = para_dict['ratio_structure_numWalks']
    ratio_delete_sequence = para_dict['ratio_delete_sequence']
    k = para_dict['k']

    # ****************************************************************************************
    # 加载图
    print("Loading graph~~~~~~~~~~~~~")
    G = load_graph(input_graph)

    start_time = time.time()
    # 计算节点的一阶和二阶相似度以及节点的信息熵
    print("Stage 1: Calculate first-order and second-order similarity and node information entropy~~~~~~~~~~~~~")
    nodes_pij_dic = cal_p_r_neibors(G, r=2)
    nodes_entropy = cal_node_entropy(G)
    end_time = time.time()
    print("         Running time: ", end_time - start_time)

    # 根据属性和拓扑采样各自所占游走次数的比例计算各自游走轮次
    ratio_attribute_numWalks = 1.0 - ratio_structure_numWalks
    structure_numWalks = total_numWalks * ratio_structure_numWalks
    attribute_numWalks = total_numWalks * ratio_attribute_numWalks

    # ****************************************************************************************
    stru_num_dict = cal_stru_num_walk(G,structure_numWalks)
    # 结构采样
    print("Stage 2: Structure sequences sampling~~~~~~~~~~~~~")
    start_time = time.time()
    walks = build_corpus(G=G, stru_num_dict=stru_num_dict, walk_length=walk_length)
    end_time = time.time()
    print("         Running time: ", end_time - start_time)
    # 属性采样
    print("Stage 3: Attribute sequences sampling~~~~~~~~~~~~~")
    # 加载样本属性
    attri_dict, num_of_features = load_attributes(input_attribute)
    attr_num_dict = cal_attr_num_walk(attri_dict,attribute_numWalks)
    start_time = time.time()
    # 构造KNN树
    print("         Build annoy (KNN) tree")
    start_time1 = time.time()
    tree = build_annoyTree(attri_dict, num_of_features)
    end_time1 = time.time()
    print("         Build annoy (KNN) tree Running time: ", end_time1 - start_time1)
    # 属性序列生成
    walks_attr, F = build_attribute_corpus(G, attr_num_dict=attr_num_dict, annoy_tree=tree, k=k, attri_dict=attri_dict, walk_length=walk_length)
    end_time = time.time()
    print("         Running time: ", end_time - start_time)

    # 对太稠密的网络可能不感冒，因为稠密的网络的随机游走多样性变大了，比较少的随机游走序列可能没有包括网络中反应局部结构和社区结构的正确的序列, 稠密网络中增大maxT 精度会上升，因为捕获得更多了，虽然也包含错误的序列，但是大部分都被我们过滤掉了。
    # 随机游走在很稠密的图上效果不好（失效，好像有文献解释，找一下）且开销要很大。

    # 按照序列所含信息量对结构序列进行筛选
    print("Stage 4: Calculate sequences entropy and refine sequences~~~~~~~~~~~~~")
    start_time = time.time()
    seq_and_seqEntropy_list = cal_sequence_entropy(walks, window_size, nodes_pij_dic, nodes_entropy)
    walks = refine_sequences(seq_and_seqEntropy_list, ratio=ratio_delete_sequence)
    end_time = time.time()
    print("         Running time: ", end_time - start_time)
    # 合并结构采样序列和属性采样序列
    walks.extend(walks_attr)
    # ************************************************************************************************************

    print("Stage 5: Trainning Skip-Gram model~~~~~~~~~~~~~")
    start_time = time.time()
    model = Word2Vec(walks, vector_size=representation_size,window=window_size, min_count=0, sg=1, hs=0, workers=workers)
    print("         Saving embedding file1~~~~~~~~~~~~~")
    model.wv.save_word2vec_format(emb_file)
    end_time = time.time()
    print("         Running time: ", end_time-start_time)

    # ******************************************************************************
    # node_id from 0

    print("Stage 6: Enhance the embedding~~~~~~~~~~~~~")
    start_time = time.time()

    R = np.loadtxt(emb_file, dtype=float, delimiter=' ', skiprows=1)
    R = R[np.argsort(R[:, 0])]  # 按第一列node id从小到大排序
    nodeID = R[:, 0]
    R = np.delete(R, 0, axis=1)  # 裁掉第一列node id

    M = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
    for k,v in nodes_pij_dic.items():
        for i,p in v.items():
            M[k][i] = p

    if R.shape[0] != M.shape[1]:
        a = abs(M.shape[1]-R.shape[0])
        R = np.row_stack((R,np.zeros((a,representation_size))))
        maxID = max(nodeID)
        for i in range(a):
            nodeID = np.insert(nodeID, [-1], values=[maxID+1+i], axis=0)

    print("         Enhancing~~~~~~~~~~~~~")
    row_norm(M)
    row_norm(F)
    for t in range(T):
        # R = R + lambda1 * np.dot(M,np.dot(F,R)) + lambda2 * np.dot(M,np.dot(F,np.dot(M,np.dot(F,R))))
        R = R + lambda1 * np.dot(F, R) + lambda2 * np.dot(M, R)
        # R = R + lambda1 * (np.dot(F, R) + np.dot(M, R))
    R = np.array([matutils.unitvec(v) for v in R])

    result = np.insert(R, 0, values=nodeID, axis=1)

    print("         Saving embedding file2~~~~~~~~~~~~~")
    with open(enhance_emb_file, 'w') as wf:
        wf.write(str(G.number_of_nodes()) + ' ' + str(representation_size) + '\n')
        for line in result:
            for index, e in enumerate(line):
                if index == 0:
                    wf.write(str(int(e)) + ' ')
                elif index == (len(line) - 1):
                    wf.write(str(round(e, 6)))
                else:
                    wf.write(str(round(e, 6)) + ' ')
            wf.write('\n')

    end_time = time.time()
    print("         Running time: ", end_time - start_time)
    # ******************************************************************************

    end_t = time.time()
    print("Total running time: ", end_t - start_t)

# 这个版本目前最好
if __name__ == '__main__':
    para_dict_list = [
                    {'graph': 'CSCW_dataset', 'emb_file': 'MS12.embeddings', 'enhance_emb_file': 'MS123.embeddings',
                    'ratio_structure_numWalks': 0.8, 'ratio_delete_sequence': 0.3, 'k': 8},
                    #   {'graph': 'Wiki', 'emb_file': 'MS1.embeddings', 'enhance_emb_file': 'MS13.embeddings',
                    # 'ratio_structure_numWalks': 1.0, 'ratio_delete_sequence': 0.7, 'k': 128},
                    #   {'graph': 'Wisconsin', 'emb_file': 'MS12-k-8.embeddings', 'enhance_emb_file': 'MS123-k-8.embeddings',
                    # 'ratio_structure_numWalks': 0.1, 'ratio_delete_sequence': 0.5, 'k': 8},
                    #   {'graph': 'Wisconsin', 'emb_file': 'MS12-k-16.embeddings', 'enhance_emb_file': 'MS123-k-16.embeddings',
                    # 'ratio_structure_numWalks': 0.1, 'ratio_delete_sequence': 0.5, 'k': 16},
                    #   {'graph': 'Wisconsin', 'emb_file': 'MS12-k-32.embeddings', 'enhance_emb_file': 'MS123-k-32.embeddings',
                    # 'ratio_structure_numWalks': 0.1, 'ratio_delete_sequence': 0.5, 'k': 32},
                    #   {'graph': 'Wisconsin', 'emb_file': 'MS12-k-64.embeddings', 'enhance_emb_file': 'MS123-k-64.embeddings',
                    # 'ratio_structure_numWalks': 0.1, 'ratio_delete_sequence': 0.5, 'k': 64},
                    #   {'graph': 'Wisconsin', 'emb_file': 'MS12-k-128.embeddings', 'enhance_emb_file': 'MS123-k-128.embeddings',
                    # 'ratio_structure_numWalks': 0.1, 'ratio_delete_sequence': 0.5, 'k': 128},
                    #   {'graph': 'Wisconsin', 'emb_file': 'MS12-k-256.embeddings', 'enhance_emb_file': 'MS123-k-256.embeddings',
                    # 'ratio_structure_numWalks': 0.1, 'ratio_delete_sequence': 0.5, 'k': 256},
                    #   {'graph': 'Wisconsin', 'emb_file': 'MS12-k-512.embeddings', 'enhance_emb_file': 'MS123-k-512.embeddings',
                    # 'ratio_structure_numWalks': 0.1, 'ratio_delete_sequence': 0.5, 'k': 512},
                    #   {'graph': 'Wisconsin', 'emb_file': 'MS12-k-1.0.embeddings', 'enhance_emb_file': 'MS123-k-1.0.embeddings',
                    # 'ratio_structure_numWalks': 0.1, 'ratio_delete_sequence': 0.5, 'k': 16}
                      ]
    for i, para_dict in enumerate(para_dict_list):
        print('Task: ', i+1)
        print('Current parameters: ', para_dict)
        main(para_dict)
