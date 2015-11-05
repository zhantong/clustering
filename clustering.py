import random
import numpy as np
import scipy.sparse.linalg
INF = 200000000  # 极大值


class Clustering():

    def __init__(self, file_name):
        """初始化
        关键变量声明，调用get_data()获取数据
        """
        self.data = []  # 存储数据集，二维数组
        self.truth = {}  # 存储行号及其label值，计算purity等需要用到
        self.file_name = file_name
        self.class_num = None  # class数目
        self.get_data()  # 初始化即调用

    def get_data(self, file_name=None, class_num=None):
        file_name = file_name or self.file_name
        with open(self.file_name, 'r') as f:
            for line_num, line in enumerate(f):
                # 每行中以“,”分界，除去最末尾的label
                t = [float(x) for x in line.split(',')[:-1]]
                self.data.append(t)
                self.truth[line_num] = line.rsplit(
                    ',', 1)[-1].strip()  # 读取每行最末尾的label
        self.class_num = class_num or len(
            set(self.truth.values()))  # class数目即不同label个数
        self.data = np.array(self.data)  # 转换为numpy.array格式
        print('class number:', self.class_num)

    def k_means(self, data=[], class_num=None):
        """k-means主体部分
        首先在data中随机挑选class_num个中心点
        计算每个中心点到所有点的距离，存储为矩阵，此时每行即点与各中心点的距离
        取离每个点最近的中心点，将其对应值全部相加取平均值后，即为新的中心点坐标
        以此循环，直到中心点坐标不再变化
        """
        if not len(data):  # 变量data的数据源
            data = self.data
        class_num = class_num or self.class_num
        # 随机挑选class_num个中心点
        classes = data[random.sample(range(len(data)), class_num)]
        while 1:
            # 共data_num行，每行为此点到各个中心点的距离；data-x即中心点坐标与所有data坐标相减，np.linalg.norm()即计算其距离
            length = np.column_stack(
                (np.linalg.norm(data-x, axis=1) for x in classes))
            args = np.argmin(length, axis=1)  # 得到length矩阵中每行最小值的下标值
            # 得到离每个中心点最近的点集合，将其坐标全部相加取平均值，此时得到新的各中心点坐标；data[np.where(args==0)[0]]即得到离中心点0最近的点的集合，(args==0).sum()即args矩阵中，值为0的元素个数
            new = np.array([np.sum(data[np.where(args == i)[0]],
                                   axis=0)/(args == i).sum() for i in range(class_num)])
            if (classes == new).all():  # 当新的中心点坐标与旧的完全相同时，循环结束
                # 计算此时所有点与其对应中心点距离的和，作为此次k-means好坏的指标
                obj_value = self.test(data, args, classes)
                return obj_value, args  # 同时还返回离每个点最近的中心点的信息，供计算purity使用
            classes = new

    def test(self, data, args, points):
        """k-means附属部分，计算所有点到其中心点距离之和
        """
        s = sum(np.linalg.norm(point-data[np.where(args == index)[0]])
                for index, point in enumerate(points))  # 每次选取一个中心点，找出其周围点，计算坐标差值，再计算其距离之和；最后将所有距离和相加
        return s

    def cal_k_means(self, data=[], class_num=None):
        """k-means控制部分
        主要控制k-means重复计算次数
        找出最优的一次计算结果
        以此计算purity和gini index
        """
        k_means_repeat_time = 10  # k-means重复计算次数
        if not len(data):  # data数据源
            data = self.data
        class_num = class_num or self.class_num
        min_value = INF  # min初值为极大值
        min_determ = None  # 保存最优解的信息
        print('k-means:')
        for i in range(k_means_repeat_time):
            value, determ = self.k_means(data, class_num)  # 调用k-means主体部分
            # value,determ=self.k_means_bak(data,class_num)
            print('第%i次: objective value: %f' % (i+1, value))
            if value < min_value:  # 如果指标更小，则作为更优解
                min_value = value
                min_determ = determ
        clus = self.cal(min_determ)  # 对最优解找出每个点在真实值和计算值的分布情况
        # clus=self.cal_bak(min_determ)
        print('选择最小值为: %f\tpurity: %f\tgini index: %f\t分布: ' % (min_value, self.purity(
            clus), self.gini_index(clus)), clus)  # 计算并输出purity和gini index

    def cal(self, args):
        """找出真实值和计算值的分布矩阵
        clus结果格式为{0:{'0':265,'1':75},1:{'0':254,'1':98}}
        外层为计算的clustering分布情况，内层为在此分类中，实际的各label值元素个数
        """
        clus = {i: {} for i in range(self.class_num)}  # 初始化clus
        for index, c in enumerate(args):
            label = self.truth[index]  # 根据行号得到其label值
            if not label in clus[c]:
                clus[c][label] = 0
            clus[c][label] += 1  # 统计同label元素个数
        return clus

    def nmf(self, data):
        """NMF主体部分
        目的是将data矩阵分解为u，v
        首先随机生成u，v矩阵
        对u，v进行迭代计算
        到达特定迭代次数后，返回u，v，和好坏指标
        """
        nmf_itera_time = 50  # NMF迭代次数
        data = np.where(data == 0, 1e-15, data)  # 将data中0全部替换为一个非常小的正数
        feature_num = len(data[0])  # feature个数
        data_num = len(data)
        # 生成u,v随机矩阵，u为feature_num*class_num; v为data_num*class_num
        u = np.random.rand(feature_num, self.class_num)
        v = np.random.rand(data_num, self.class_num)
        x = data.T  # data矩阵转置
        while nmf_itera_time:
            u = u * np.dot(x, v) / np.dot(np.dot(u, v.T), v)  # 对u, v迭代计算
            v = v * np.dot(x.T, u) / np.dot(np.dot(v, u.T), u)
            nmf_itera_time -= 1
        t = x-np.dot(u, v.T)  # 临时变量，原矩阵与分解后的矩阵的积的坐标差值
        j = np.sum(t*t)  # 计算距离，实际上作为判断此次NMF好坏的指标
        return j, u, v

        self.cal_nmf(v)

    def cal_nmf(self, data):
        """NMF的控制部分
        主要控制NMF重复计算次数
        找出最优的一次计算结果
        以此计算purity和gini index
        """
        nmf_repeat_time = 10  # NMF重复计算次数
        min_j = INF  # min初值为极大值
        min_u = None  # 临时保存
        min_v = None
        print('NMF:')
        for i in range(nmf_repeat_time):
            j, u, v = self.nmf(data)  # 调用NMF主体部分
            print('第%i次，objective value: %f' % (i+1, j))
            if j < min_j:  # 寻找最小值即最优解
                min_j = j
                min_u = u
                min_v = v
        u_s = np.sum(min_u * min_u, axis=0)**0.5
        t = np.tile(u_s, (len(data), 1))
        min_v = min_v * t
        maxs = np.argmax(min_v, axis=1)  # 得到v中每行最大值的下标
        clus = self.cal(maxs)  # 对最优解找出每个点在真实值和计算值的分布情况
        print('选择最小值为: %f\tpurity: %f\tgini index: %f' % (
            min_j, self.purity(clus), self.gini_index(clus)))  # 计算并输出purity和gini index

    def spectral(self, data, n):
        """Spectral Algorism主体部分
        主要思想即降维
        找出每个点与其最近的n个邻居
        再通过矩阵运算，得到存储有其邻居信息的矩阵
        对此矩阵求特征值和特征向量
        选取最小的class_num个特征值的特征向量
        将特征向量组成的矩阵进行k-means计算
        即得到clustering的结果
        """
        data_num = len(data)
        # 虽然w值全是整数，但为了求特征值方便，声明为浮点
        w = np.zeros((data_num, data_num), dtype='float')
        for row in range(data_num):  # 构建图
            if row % 100 == 0:  # 每处理100行输出一次提示信息
                print('已处理%i/%i' % (row, data_num))
            dist = np.linalg.norm(data-data[row], axis=1)  # 计算此点与其他所有点的距离
            # 对于前n+1（因为这个点与自己的距离最短，所以加1）个邻居，在新的矩阵里赋值-1
            for column in np.argsort(dist)[:n+1]:
                w[row][column], w[column][row] = -1, -1  # 赋值为-1可以减少一个矩阵运算
        for row in range(data_num):  # 对应spectral algorism中矩阵相减
            w[row][row] = -np.sum(w[row]) - 1
        print('计算特征值与特征向量...')
        # 这里采用scipy专门用于计算稀疏矩阵特征值特征向量的方法，且可以指定计算的特征值个数，这里限定为特征值从小到达排，计算前class_num个
        r, v = scipy.sparse.linalg.eigsh(w, k=self.class_num + 1, which='SA')
#		r, v = np.linalg.eig(w)
#		print('2')
#		for arg in np.argsort(r)[:self.class_num]:
#			res.append(v[arg])
#		print('3')
#		res = np.dstack(res)[0]
#		print('4')
        res = v[1:]
        return res  # 返回class_num个特征向量

    def cal_spectral(self, data):
        """spectral algorism的控制部分
        对指定的邻居个数，调用spectral()得到特征向量
        再对特征向量进行k-means
        """
        neighbors = [3, 6, 9]  # 邻居个数的list
        print('Spectral Clustering:')
        for n in neighbors:
            print('nearest neighbors n: %i' % n)
            res = self.spectral(data, n)
            self.cal_k_means(res)

    def purity(self, clus):
        """计算purity指标
        根据clus得到的矩阵，计算purity
        """
        sum_p = 0
        sum_m = 0
        for classes in clus.values():
            if classes:
                sum_p += max(classes.values())
                sum_m += sum(classes.values())
        p = sum_p / sum_m
        return p

    def gini_index(self, clus):
        """计算gini index指标
        根据clus得到的矩阵，计算gini index
        """
        sum_gm = 0
        sum_m = 0
        for classes in clus.values():
            if classes:
                m = sum(classes.values())
                g = 1
                for c in classes.values():
                    g -= (c / m)**2
                sum_gm += g * m
                sum_m += m
        g_avg = sum_gm / sum_m
        return g_avg

    def start_all(self):
        """顺序开始三个计算
        """
        self.cal_k_means(self.data)
        print('-'*20)
        self.cal_nmf(self.data)
        print('-'*20)
        self.cal_spectral(self.data)
        print('all done!')

# 以下为老代码

    def k_means_bak(self, data=[], class_num=None):
        if not len(data):
            data = self.data
        class_num = class_num or self.class_num
        feature_num = len(data[0])
        classes = data[random.sample(range(len(data)), class_num)]
        determ = {}
        while 1:
            for i in range(class_num):
                determ[i] = []
            new = np.zeros((class_num, feature_num))
            count = np.zeros((class_num, 1), dtype='int')
            for line_num, item in enumerate(data):
                the_min = INF
                min_index = -1
                for index, c in enumerate(classes):
                    t = item - c
                    dist = np.sum(t * t)
                    if dist < the_min:
                        the_min = dist
                        min_index = index
                determ[min_index].append(line_num)
                new[min_index] += item
                count[min_index][0] += 1
            new = new / count
            if (classes == new).all():
                obj_value = self.cal_obj_value(data, determ, classes)
                return obj_value, determ
            classes = new
            # print(count)
            #print([len(x) for x in determ.values()])

    def cal_obj_value(self, data, determ, points):
        the_sum = 0
        for index, classes in determ.items():
            point = points[index]
            for c in classes:
                t = data[c] - point
                the_sum += np.sum(t * t)
        return the_sum

    def cal_bak(self, determ):
        clus = {}
        for index, classes in determ.items():
            clus[index] = {}
            for c in classes:
                label = self.truth[c]
                if not label in clus[index]:
                    clus[index][label] = 0
                clus[index][label] += 1
        return clus

    def spectral_bak(self, data, n):
        data_num = len(data)
        dists = np.zeros((data_num, data_num))
        w = np.zeros((data_num, data_num), dtype='int')
#		global q
#		for row in range(data_num):
#			q.put(row)
#		self.multi_thread(30,self.cal_matrix,data,dists,data_num)

        for row in range(data_num):
            # print(row)
            for column in range(row, data_num):
                t = data[row] - data[column]
                s = np.sum(t * t)
                dists[row][column], dists[column][row] = s, s
        sort_all = np.argsort(dists)
        for row in range(data_num):
            for arg in sort_all[row][:n + 1]:
                w[row][arg] = -1
                w[arg][row] = -1
        for row in range(data_num):
            w[row][row] = -np.sum(w[row]) - 1
        res = []
        r, v = np.linalg.eig(w)
        for arg in np.argsort(r)[:self.class_num]:
            res.append(v[arg])
        res = np.dstack(res)[0]
        return res
        #self.cal_k_means(res, self.class_num)
if __name__ == '__main__':
    c=Clustering('german.txt')
    c.start_all()
    c = Clustering('mnist.txt')
    c.start_all()
    # c.cal_k_means()
    # c.cal_nmf(c.data)
    # c.cal_spectral(c.data)
