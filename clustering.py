import random
import numpy as np
import time
import threading


class Clustering():

	def __init__(self):
		self.data = []
		self.truth = {}
		self.get_data()

	def get_data(self):
		class_num = set()
		last = []
		line_num = 0
		with open('mnist.txt', 'r') as f:
			for line in f:
				t = [float(x) for x in line.split(',')]
				self.data.append(t[:-1])
				self.truth[line_num] = line.rsplit(',', 1)[-1].strip()
				line_num += 1
		# print(self.truth)
		self.feature_num = len(self.data[0])
		self.class_num = len(set(self.truth.values()))
		self.data_num = len(self.data)
		print('class number:', self.class_num)
		self.data = np.array(self.data)
		class_set = sorted(set(last))
		for class_name in class_set:
			print('%i:%i' % (class_name, last.count(class_name)), end='\t')
		# print('1:%i\t-1:%i'%(last.count('1'),last.count('-1')))
		# print(class_num)

	def get_random(self, length, min, max):
		return [random.uniform(min, max) for x in range(length)]

	def k_means(self, data, class_num):
		feature_num = len(data[0])
		the_max = np.max(data)
		the_min = np.min(data)
		print('max:%f\tmin:%f' % (the_max, the_min))
		classes = [self.get_random(feature_num, the_min, the_max)
					for x in range(class_num)]
		classes = np.array(classes)
		determ = {}

		while 1:
			for i in range(class_num):
				determ[i] = []
			new = np.zeros((class_num, feature_num))
			count = np.zeros((class_num, 1), dtype='int')
			for line_num, item in enumerate(data):
				min = 200000000
				min_index = -1
				for index, c in enumerate(classes):
					t = item - c
					dist = np.sum(t * t)
					if dist < min:
						min = dist
						min_index = index
				determ[min_index].append(line_num)
				new[min_index] += item
				count[min_index][0] += 1
			zeros = []
			for index in range(len(count)):
				if not count[index][0]:
					zeros.append(index)
					new[index] = classes[index]
					# print(new[index])
					count[index][0] = 1
			new = new / count
			# print(new[zeros[0]])
			if (classes == new).all():
				print(self.cal_obj_value(data, determ, classes))
				print(self.cal(determ))
				print(self.purity(self.cal(determ)))
				print(self.gini_index(self.cal(determ)))
				break
			classes = new
			# print(count)
			print([len(x) for x in determ.values()])

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
			# print(sum_p,sum_m)
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
		the_max = np.max(data)
		the_min = np.min(data)
		print('max:%f\tmin:%f' % (the_max, the_min))
		u = [self.get_random(self.class_num, the_min + 0.01, the_max)
					for row in range(self.feature_num)]
		print(u[0])
		print(u[1])
		v = [self.get_random(self.class_num, the_min + 0.01, the_max)
					for row in range(self.data_num)]
		u = np.array(u)
		v = np.array(v)
		x = data.T
		n = 50
		while n:
			# u=u*(x*v)/(u*v.T*v)
			# v=v*(x.T*u)/(v*u.T*u)

			last_u = u
			u = u * np.dot(x, v) / np.dot(np.dot(u, v.T), v)
			v = v * np.dot(x.T, u) / np.dot(np.dot(v, u.T), u)
			if (last_u == u).all():
				print('OK')
			# print(n)
			n -= 1
		u_s = np.sum(u * u, axis=0)**0.5
		t = np.tile(u_s, (self.data_num, 1))
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
		print(clus)
		print(self.purity(clus))
		print(self.gini_index(clus))
		count = [0] * self.class_num
		for arg in maxs:
			count[arg] += 1
		print(count)

	def spectral(self, n,start,end):
		data = self.data
		data_num = len(data)
		print(data_num)
		dists = np.zeros((data_num, data_num))
		w = np.zeros((data_num, data_num), dtype='int')
#		for row in range(data_num):
#			print(row)
#			stack=[]
#			for column in range(n+1):
#				t=data[row]-data[column]
#				s=np.sum(t*t)
#				stack.append(s)
#			stack=sorted(stack)
#			for column in range(n+1,data_num):
#				t=data[row]-data[column]
#				t=sorted(t,reverse=True)
#				the_sum=0
#				flag=1
#				for k in t[:100]:
#					the_sum+=k*k
#				if the_sum>=stack[-1]:
# print('continue')
#					continue
#				for k in t[100:]:
#					the_sum+=k*k
#					if the_sum>=stack[-1]:
#						flag=0
#						break
#				if flag:
#					stack[-1]=the_sum
#					stack=sorted(stack)
#		return
		for row in range(start,end):
			stack = []
			print(row)
			for column in range(n + 1):
				t = data[row] - data[column]
				s = np.sum(t * t)
				stack.append(s)
			stack = sorted(stack)
			for column in range(n + 1, data_num):
				t = data[row] - data[column]
				d = np.sort(t)[-100:]
				if np.sum(d * d) >= stack[-1]:
					continue
				s = np.sum(t * t)
				if s >= stack[-1]:
					continue
				else:
					stack[-1] = s
					stack = sorted(stack)

		for row in range(data_num):
			print(row)
			for column in range(row, data_num):
				t = data[row] - data[column]
				s = np.sum(t * t)
				dists[row][column], dists[column][row] = s, s
		print('1')
		sort_all = np.argsort(dists)
		print('2')
		for row in range(data_num):
			for arg in sort_all[row][:n + 1]:
				w[row][arg] = -1
				w[arg][row] = -1
		print('3')
		for row in range(data_num):
			w[row][row] = -np.sum(w[row]) - 1
		res = []
		r, v = np.linalg.eig(w)
		for arg in np.argsort(r)[:self.class_num]:
			res.append(v[arg])
		res = np.dstack(res)[0].tolist()
		self.k_means(res, self.class_num)


	def multi_thread(self, num, target):  # 多线程模板
		threads = []
		n=0
		for i in range(num):
			d = threading.Thread(target=target,args=(3,n,n+1000))
			n+=1000
			threads.append(d)
		for d in threads:
			d.start()
		for d in threads:
			d.join()






if __name__=='__main__':
	c=Clustering()
	# c.k_means(c.data,c.class_num)
	# c.nmf(c.data)
	#c.spectral(3)
	c.multi_thread(10,c.spectral)
