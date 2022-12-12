import warnings

import numpy as np
import time
import geatpy as ea  # 导入geatpy库

class moea_multi_NSGA2(ea.MoeaAlgorithm):
    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, dirName)
        if type(population) != list:
            raise RuntimeError('传入的种群对象列表必须为list类型')
        if type(problem) != list:
            raise RuntimeError('传入的问题对象列表必须为list类型')
        self.PopNum = len(population)  # 种群数目
        self.name = 'multi_NSGA2'
        if self.problem[0].M < 10:
            self.ndSort = ea.ndsortESS  # 采用ENS_SS进行非支配排序
        else:
            self.ndSort = ea.ndsortTNS  # 高维目标采用T_ENS进行非支配排序，速度一般会比ENS_SS要快
        self.selFunc = 'tour'  # 选择方式，采用锦标赛选择
        self.migFr = 5  # 发生种群迁移的间隔代数

        if population[0].Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)  # 生成部分匹配交叉算子对象
            self.mutOper = ea.Mutinv(Pm=1)  # 生成逆转变异算子对象
        elif population[0].Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)  # 生成均匀交叉算子对象
            self.mutOper = ea.Mutbin(Pm=None)  # 生成二进制变异算子对象，Pm设置为None时，具体数值取变异算子中Pm的默认值
        elif population[0].Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)  # 生成模拟二进制交叉算子对象
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)  # 生成多项式变异算子对象
        else:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')

    def unite(self, population):
        """
        合并种群，生成联合种群。
        注：返回的unitePop不携带Field和Chrom的信息，因为其Encoding=None。
        """
        # 遍历种群列表，构造联合种群
        unitePop = ea.Population(None, None, population[0].sizes, None,  # 第一个输入参数传入None，设置Encoding为None
                                 ObjV=population[0].ObjV,
                                 FitnV=population[0].FitnV,
                                 CV=population[0].CV,
                                 Phen=population[0].Phen)
        for i in range(1, self.PopNum):
            unitePop += population[i]
        return unitePop

    def reinsertion(self, population, offspring, NUM, popNum):

        """
        描述:
            重插入个体产生新一代种群（采用父子合并选择的策略）。
            NUM为所需要保留到下一代的个体数目。
            注：这里对原版NSGA-II进行等价的修改：先按帕累托分级和拥挤距离来计算出种群个体的适应度，
            然后调用dup选择算子(详见help(ea.dup))来根据适应度从大到小的顺序选择出个体保留到下一代。
            这跟原版NSGA-II的选择方法所得的结果是完全一样的。

        """

        # 父子两代合并
        population = population + offspring
        # 选择个体保留到下一代
        [levels, _] = self.ndSort(population.ObjV, NUM, None, population.CV, self.problem[popNum].maxormins)  # 对NUM个个体进行非支配分层
        dis = ea.crowdis(population.ObjV, levels)  # 计算拥挤距离
        population.FitnV[:, 0] = np.argsort(np.lexsort(np.array([dis, -levels])), kind='mergesort')  # 计算适应度
        chooseFlag = ea.selecting('dup', population.FitnV, NUM)  # 调用低级选择算子dup进行基于适应度排序的选择，保留NUM个个体
        return population[chooseFlag]

    def call_aimFunc(self, pop, popNum):

        """
        描述: 调用问题类的aimFunc()或evalVars()完成种群目标函数值和违反约束程度的计算。

        例如：population为一个种群对象，则调用call_aimFunc(population)即可完成目标函数值的计算。
             之后可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。

        输入参数:
            pop : class <Population> - 种群对象。

        输出参数:
            无输出参数。

        """

        pop.Phen = pop.decoding()  # 染色体解码
        if self.problem[popNum] is None:
            raise RuntimeError('error: problem has not been initialized. (算法类中的问题对象未被初始化。)')
        self.problem[popNum].evaluation(pop)  # 调用问题类的evaluation()
        self.evalsNum = self.evalsNum + pop.sizes if self.evalsNum is not None else pop.sizes  # 更新评价次数
        # 格式检查
        if not isinstance(pop.ObjV, np.ndarray) or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or \
                pop.ObjV.shape[1] != self.problem[popNum].M:
            raise RuntimeError('error: ObjV is illegal. (目标函数值矩阵ObjV的数据格式不合法，请检查目标函数的计算。)')
        if pop.CV is not None:
            if not isinstance(pop.CV, np.ndarray) or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
                raise RuntimeError('error: CV is illegal. (违反约束程度矩阵CV的数据格式不合法，请检查CV的计算。)')

    def logging(self, pop, popNum=0):

        """
        描述:
            用于在进化过程中记录日志。该函数在stat()函数里面被调用。
            如果需要在日志中记录其他数据，需要在自定义算法类中重写该函数。

        输入参数:
            pop : class <Population> - 种群对象。

        输出参数:
            无输出参数。

        """

        self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算logging的耗时
        if len(self.log['gen']) == 0:  # 初始化log的各个键值
            self.log['gd'] = []
            self.log['igd'] = []
            self.log['hv'] = []
            self.log['spacing'] = []
        self.log['gen'].append(self.currentGen)
        self.log['eval'].append(self.evalsNum)  # 记录评价次数
        [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV, maxormins=self.problem[popNum].maxormins)  # 非支配分层
        NDSet = pop[np.where(levels == 1)[0]]  # 只保留种群中的非支配个体，形成一个非支配种群
        if self.problem[popNum].ReferObjV is not None:
            self.log['gd'].append(ea.indicator.GD(NDSet.ObjV, self.problem[popNum].ReferObjV))  # 计算GD指标
            self.log['igd'].append(ea.indicator.IGD(NDSet.ObjV, self.problem[popNum].ReferObjV))  # 计算IGD指标
            self.log['hv'].append(ea.indicator.HV(NDSet.ObjV), )  # 计算HV指标
        else:
            self.log['gd'].append(None)
            self.log['igd'].append(None)
            self.log['hv'].append(ea.indicator.HV(NDSet.ObjV))  # 计算HV指标
        self.log['spacing'].append(ea.indicator.Spacing(NDSet.ObjV))  # 计算Spacing指标
        self.timeSlot = time.time()  # 更新时间戳

    def stat(self, pop, popNum=0):

        """
        描述:
            该函数用于分析当代种群的信息。
            该函数会在terminated()函数里被调用。

        输入参数:
            pop : class <Population> - 种群对象。

        输出参数:
            无输出参数。

        """

        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  # 找到满足约束条件的个体的下标
        if len(feasible) > 0:
            feasiblePop = pop[feasible]  # 获取满足约束条件的个体
            if self.logTras != 0 and self.currentGen % self.logTras == 0:
                self.logging(feasiblePop, popNum=popNum)  # 记录日志
                if self.verbose:
                    self.display()  # 打印日志
            self.draw(feasiblePop,popNum=popNum)  # 展示输出


    def terminated(self, pop, popNum=0):

        """
        描述:
            该函数用于判断是否应该终止进化，population为传入的种群对象。
            该函数会在各个具体的算法类的run()函数中被调用。

        输入参数:
            pop : class <Population> - 种群对象。

        输出参数:
            True / False。

        """
        for i in range(self.PopNum):
            self.check(pop[i])  # 检查种群对象的关键属性是否有误
            self.stat(pop[i], i)  # 进行统计分析
            self.passTime += time.time() - self.timeSlot  # 更新耗时
        # 调用outFunc()
        if self.outFunc is not None:
            if not callable(self.outFunc):
                raise RuntimeError('outFunc must be a function. (如果定义了outFunc，那么它必须是一个函数。)')
            self.outFunc(self, pop)
        self.timeSlot = time.time()  # 更新时间戳
        # 判断是否终止进化，
        if self.MAXGEN is None and self.MAXTIME is None and self.MAXEVALS is None:
            raise RuntimeError(
                'error: MAXGEN, MAXTIME, and MAXEVALS cannot be all None. (MAXGEN, MAXTIME, 和MAXEVALS不能全为None)')
        terminatedFlag = False
        if self.MAXGEN is not None and self.currentGen + 1 >= self.MAXGEN:  # 由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
            self.stopMsg = 'The algotirhm stepped because it exceeded the generation limit.'
            terminatedFlag = True
        if self.MAXTIME is not None and self.passTime >= self.MAXTIME:
            self.stopMsg = 'The algotirhm stepped because it exceeded the time limit.'
            terminatedFlag = True
        if self.MAXEVALS is not None and self.evalsNum >= self.MAXEVALS:
            self.stopMsg = 'The algotirhm stepped because it exceeded the function evaluation limit.'
            terminatedFlag = True
        if terminatedFlag:
            return True
        else:
            self.currentGen += 1  # 进化代数+1
            return False

    def draw(self, pop, popNum=0, EndFlag=False):

        """
        描述:
            该函数用于在进化过程中进行绘图。该函数在stat()以及finishing函数里面被调用。

        输入参数:
            pop     : class <Population> - 种群对象。

            EndFlag : bool - 表示是否是最后一次调用该函数。

        输出参数:
            无输出参数。

        """

        if not EndFlag:
            self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算画图的耗时
            # 绘制动画
            if self.drawing == 2:
                # 绘制目标空间动态图
                if pop.ObjV.shape[1] == 2 or pop.ObjV.shape[1] == 3:
                    if self.plotter is None:
                        self.plotter = ea.PointScatter(self.problem[popNum].M, grid=True, legend=True,
                                                       title='Pareto Front Plot')
                    self.plotter.refresh()
                    self.plotter.add(pop.ObjV, color='red', label='MOEA PF at ' + str(self.currentGen) + ' Generation')
                else:
                    if self.plotter is None:
                        self.plotter = ea.ParCoordPlotter(self.problem[popNum].M, grid=True, legend=True,
                                                          title='Parallel Coordinate Plot')
                    self.plotter.refresh()
                    self.plotter.add(pop.ObjV, color='red',
                                     label='MOEA Objective Value at ' + str(self.currentGen) + ' Generation')
                self.plotter.draw()
            elif self.drawing == 3:
                # 绘制决策空间动态图
                if self.plotter is None:
                    self.plotter = ea.ParCoordPlotter(self.problem[popNum].Dim, grid=True, legend=True,
                                                      title='Variables Value Plot')
                self.plotter.refresh()
                self.plotter.add(pop.Phen, marker='o', color='blue',
                                 label='Variables Value at ' + str(self.currentGen) + ' Generation')
                self.plotter.draw()
            self.timeSlot = time.time()  # 更新时间戳
        else:
            # 绘制最终结果图
            if self.drawing != 0:
                if self.plotter is not None:  # 若绘制了动画，则保存并关闭动画
                    self.plotter.createAnimation()
                    self.plotter.close()
                if pop.ObjV.shape[1] == 2 or pop.ObjV.shape[1] == 3:
                    figureName = 'Pareto Front Plot'
                    self.plotter = ea.PointScatter(self.problem[popNum].M, grid=True, legend=True, title=figureName,
                                                   saveName=self.dirName + figureName)
                    self.plotter.add(self.problem[popNum].ReferObjV, color='gray', alpha=0.1, label='True PF')
                    self.plotter.add(pop.ObjV, color='red', label='MOEA PF')
                    self.plotter.draw()
                else:
                    figureName = 'Parallel Coordinate Plot'
                    self.plotter = ea.ParCoordPlotter(self.problem[popNum].M, grid=True, legend=True, title=figureName,
                                                      saveName=self.dirName + figureName)
                    self.plotter.add(self.problem[popNum].TinyReferObjV, color='gray', alpha=0.5, label='True Objective Value')
                    self.plotter.add(pop.ObjV, color='red', label='MOEA Objective Value')
                    self.plotter.draw()

    def finishing(self, pop, popNum=0, globalNDSet=None):

        """
        描述:
            进化完成后调用的函数。

        输入参数:
            pop : class <Population> - 种群对象。

            globalNDSet : class <Population> - (可选参数)全局存档。

        输出参数:
            [NDSet, pop]，其中pop为种群类型；NDSet的类型与pop的一致。

        """
        if globalNDSet is None:
            # 得到非支配种群
            [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV, maxormins=self.problem[popNum].maxormins)  # 非支配分层
            NDSet = pop[np.where(levels == 1)[0]]  # 只保留种群中的非支配个体，形成一个非支配种群
            if NDSet.CV is not None:  # CV不为None说明有设置约束条件
                NDSet = NDSet[np.where(np.all(NDSet.CV <= 0, 1))[0]]  # 最后要彻底排除非可行解
        else:
            NDSet = globalNDSet
        if self.logTras != 0 and NDSet.sizes != 0 and (
                len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen):  # 补充记录日志和输出
            self.logging(NDSet, popNum)
            if self.verbose:
                self.display()
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，因为已经要结束，因此不用再更新时间戳
        self.draw(NDSet, popNum=popNum, EndFlag=True)  # 显示最终结果图
        if self.plotter is not None:
            self.plotter.show()
        # 返回帕累托最优个体以及最后一代种群
        return [NDSet, pop]

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        for i in range(self.PopNum):  # 遍历每个种群，初始化每个种群的染色体矩阵
            NIND = population[i].sizes
            population[i].initChrom(population[i].sizes)  # 初始化种群染色体矩阵
            # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）暂时用不到
            if prophetPop is not None:
                population[i] = (prophetPop[i] + population[i])[:NIND]  # 插入先知种群
            self.call_aimFunc(population[i],i)  # 计算种群的目标函数值

            [levels, _] = self.ndSort(population[i].ObjV, NIND, None, population[i].CV, self.problem[i].maxormins)  # 对NIND个个体进行非支配分层
            population[i].FitnV = (1 / levels).reshape(-1, 1)  # 直接根据levels来计算初代个体的适应度
        # ===========================开始进化============================
        iter = 0
        while True:
            if self.terminated(population) == True:
                break;
            for i in range(self.PopNum):
                # 选择个体参与进化
                offspring = population[i][ea.selecting(self.selFunc, population[i].FitnV, NIND)]
                # 对选出的个体进行进化操作
                offspring.Chrom = self.recOper.do(offspring.Chrom)  # 重组
                # if self.currentGen > self.MAXGEN * 0.5:
                #     offspring.Chrom = ea.mutmani(offspring.Encoding, offspring.Chrom, offspring.Field, self.problem.M-1)
                offspring.Chrom = self.mutOper.do(offspring.Encoding, offspring.Chrom, offspring.Field)  # 变异
                self.call_aimFunc(offspring, i)  # 求进化后个体的目标函数值
                population[i] = self.reinsertion(population[i], offspring, NIND, i)  # 重插入生成新一代种群
                if iter%20 ==0 and iter !=0 :
                    for j in range(self.PopNum):
                        if(j!=i):
                            temp = population[j].copy()
                            self.call_aimFunc(temp, i)
                            population[i] = self.reinsertion(population[i], temp, NIND, i)  # 重插入生成新一代种群
            iter += 1
        res = []
        for i in range(self.PopNum):
            res.append(self.finishing(population[i], popNum=i))

        return res  # 调用finishing完成后续工作并返回结果