import random
import numpy as np
import networkx as nx
import ReadData as rd
import Locus_base.LocusBase_tools as locus_base

#输入为两个list，输出也是两个list
# 两个个体交叉更新
def crossover(list1,list2,index):
    child1=list1[0:index]+list2[index:]
    child2 = list2[0:index] + list1[index:]
    return child1,child2

# 两个种群之间的交叉操作
def crossover_population(list1,list2,index):
    return None

def mutation():
    return None

# 种群整体的更新策略
def update_population(population):
    res=[]
    return res

# 单个个体的更新操作
def update_individual(individual):
    res=[]
    return res
