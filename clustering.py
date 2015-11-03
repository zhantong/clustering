import random
import numpy as np
import time
import threading
import queue
q=queue.Queue()
mutex=threading.Lock()
INF=200000000
class Clustering():

	def __init__(self,file_name):
		self.data = []
		self.truth = {}
		self.file_name=file_name
		self.get_data()

	def get_data(self,file_name=None,class_num=None):
		file_name=file_name or self.file_name
		with open(self.file_name, 'r') as f:
			for line_num,line in enumerate(f):
				t = [float(x) for x in line.split(',')]
				self.data.append(t[:-1])
				self.truth[line_num] = line.rsplit(',', 1)[-1].strip()
		self.class_num = class_num or len(set(self.truth.values()))
		self.data=np.array(self.data)
		print('class number:', self.class_num)

	def k_means(self, data=[], class_num=None):
		if not len(data):
			data=self.data
		class_num=class_num or self.class_num
		feature_num = len(data[0])
		classes=data[random.sample(range(len(data)),class_num)]
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
				obj_value=self.cal_obj_value(data, determ, classes)
				return obj_value,determ
			classes = new
			# print(count)
			#print([len(x) for x in determ.values()])
	def cal_k_means(self,data=[],class_num=None):
		if not len(data):
			data=self.data
		class_num=class_num or self.class_num
		min_value=INF
		min_determ=None
		print('k-means:')
		for i in range(10):
			value,determ=self.k_means(data,class_num)
			print('第%i次，objective value: %f'%(i+1,value))
			if value<min_value:
				min_value=value
				min_determ=determ
		clus=self.cal(min_determ)
		print('选择最小值为: %f\tpurity: %f\tgini index: %f'%(min_value,self.purity(clus),self.gini_index(clus)))
		#print(self.purity(clus))
		#print(self.gini_index(clus))

	def cal_obj_value(self, data, determ, points):
		the_sum = 0
		for index, classes in determ.items():
			point = points[index]
			for c in classes:
				t = data[c] - point
				the_sum += np.sum(t * t)
		return the_sum

	def cal(self, determ):
		clus = {}
		for index, classes in determ.items():
			clus[index] = {}
			for c in classes:
				label = self.truth[c]
				if not label in clus[index]:
					clus[index][label] = 0
				clus[index][label] += 1
		return clus

	def purity(self, clus):
		sum_p = 0
		sum_m = 0
		for classes in clus.values():
			if classes:
				sum_p += max(classes.values())
				sum_m += sum(classes.values())
		p = sum_p / sum_m
		return p

	def gini_index(self, clus):
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

	def nmf(self, data):
		data = np.where(data == 0, 1e-10, data)
		feature_num=len(data[0])
		data_num=len(data)
		u = [[random.random() for column in range(self.class_num)] for row in range(feature_num)]
		v = [[random.random() for column in range(self.class_num)] for row in range(data_num)]
		u = np.array(u)
		v = np.array(v)
		x = data.T
		n = 50
		while n:
			u = u * np.dot(x, v) / np.dot(np.dot(u, v.T), v)
			v = v * np.dot(x.T, u) / np.dot(np.dot(v, u.T), u)
			n -= 1
		t=x-np.dot(u,v.T)
		j=np.sum(t*t)
		return j,u,v

		self.cal_nmf(v)
	def do_nmf(self,u,v,data_num):
		u_s = np.sum(u * u, axis=0)**0.5
		t = np.tile(u_s, (data_num, 1))
		v = v * t		
		maxs = np.argmax(v, axis=1)
		clus = {}
		for i in range(self.class_num):
			clus[i] = {}
		for index, c in enumerate(maxs):
			label = self.truth[index]
			if not label in clus[c]:
				clus[c][label] = 0
			clus[c][label] += 1
		return clus
		print(clus)
		print(self.purity(clus))
		print(self.gini_index(clus))
		count = [0] * self.class_num
		for arg in maxs:
			count[arg] += 1
		print(count)
	def cal_nmf(self,data):
		min_j=INF
		min_u=None
		min_v=None
		print('NMF:')
		for i in range(10):
			j,u,v=self.nmf(data)
			print('第%i次，objective value: %f'%(i+1,j))
			if j<min_j:
				min_j=j
				min_u=u
				min_v=v
		clus=self.do_nmf(min_u,min_v,len(data))
		print('选择最小值为: %f\tpurity: %f\tgini index: %f'%(min_j,self.purity(clus),self.gini_index(clus)))
		#print(self.purity(clus))
		#print(self.gini_index(clus))
	def cal_matrix(self,data,dists,data_num):
		global q
		while not q.empty():
			row=q.get()
			for i in range(data_num):
				t = data[row] - data[i]
				s = np.sum(t * t)
				dists[row][i], dists[i][row] = s, s
	def spectral(self, data,n):
		data_num = len(data)
		dists = np.zeros((data_num, data_num))
		w = np.zeros((data_num, data_num), dtype='int')
#		global q
#		for row in range(data_num):
#			q.put(row)
#		self.multi_thread(30,self.cal_matrix,data,dists,data_num)

		for row in range(data_num):
			#print(row)
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
	def cal_spectral(self,data):
		print('Spectral Clustering:')
		for n in [3,6,9]:
			print('nearest neighbors n: %i'%n)
			res=self.spectral(data,n)
			self.cal_k_means(res)

	def multi_thread(self, num, target,data,dists,data_num):  # 多线程模板
		threads = []
		for i in range(num):
			d = threading.Thread(target=target,args=(data,dists,data_num))
			threads.append(d)
		for d in threads:
			d.start()
		for d in threads:
			d.join()

	def start_all(self):
		self.cal_k_means(self.data)
		print('-'*20)
		self.cal_nmf(self.data)
		print('-'*20)
		self.cal_spectral(self.data)
		print('all done!')




if __name__=='__main__':
	c=Clustering('german.txt')
	c.start_all()
	#c.cal_k_means()
	#c.cal_nmf(c.data)
	#c.cal_spectral(c.data)
	#c.multi_thread(10,c.spectral)
