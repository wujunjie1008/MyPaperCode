def merge_subsets(sub):
    arr = []
    to_skip = []
    for s in range(len(sub)):
        if sub[s] not in to_skip:
            new = sub[s]
            for x in sub:
                if sub[s] & x:
                    new = new | x
                    to_skip.append(x)
            arr.append(new)
    return arr

# locus-base的解码
def decode(chrom):
    # sub为一个list，list中的每个元素为一个set集合,例如{0,1}，表示一个社区
    sub = [{x, chrom[x]} for x in range(len(chrom))]
    result = sub
    i = 0
    while i < len(result):
        candidate = merge_subsets(result)
        if candidate != result:
            result = candidate
        else:
            break
        result = candidate
        i += 1
    for i in range(len(result)):
        result[i] = list(result[i])
    return result

# data = [1, 0, 3, 1, 5, 4, 5, 10, 9, 7, 7]
# print(decode(data))