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