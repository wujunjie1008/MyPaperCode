import numpy as np
from collections import defaultdict

class Alias:
    def __init__(self):
        self.alias = defaultdict(list)
        self.accept = defaultdict(list)
        self.map_id2node = defaultdict(dict)
        # self.node_map =

    def create_alias_table(self, center_node, area_node, area_ratio):

        N = len(area_ratio)
        self.accept[center_node], self.alias[center_node] = [0] * N, [0] * N
        small, large = [], []
        area_ratio_ = np.array(area_ratio)

        # ratio_max = np.max(area_ratio_)
        # area_ratio_ = np.exp(area_ratio_ - ratio_max) /\
        # np.sum(np.exp(area_ratio_ - ratio_max))

        sum_ratio = np.sum(area_ratio_)
        area_ratio_ = N * area_ratio_ / sum_ratio


        # print(area_node)
        # print(area_ratio)
        # print(area_ratio_)


        for i, prob in enumerate(area_ratio_):
            self.map_id2node[center_node][i] = area_node[i]
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            self.accept[center_node][small_idx] = area_ratio_[small_idx]
            self.alias[center_node][small_idx] = large_idx
            area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                     (1 - area_ratio_[small_idx])
            if area_ratio_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            self.accept[center_node][large_idx] = 1
        while small:
            small_idx = small.pop()
            self.accept[center_node][small_idx] = 1

        # return accept, alias

    def alias_sample(self, center_node):

        N = len(self.accept[center_node])
        i = int(np.random.random() * N)
        r = np.random.random()

        if r < self.accept[center_node][i]:
            chosen_id = i
        else:
            chosen_id = self.alias[center_node][i]

        return self.map_id2node[center_node][chosen_id]

# def softmax(x, axis=1):
#     # ????????????????????????
#     row_max = x.max(axis=axis)
#
#     # ?????????????????????????????????????????????????????????exp(x)??????????????????inf??????
#     row_max = row_max.reshape(-1, 1)
#     x = x - row_max
#
#     # ??????e???????????????
#     x_exp = np.exp(x)
#     x_sum = np.sum(x_exp, axis=axis, keepdims=True)
#     s = x_exp / x_sum
#     return s





