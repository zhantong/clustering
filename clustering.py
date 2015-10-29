import random
import numpy as np
import time
class Clustering():
	def __init__(self):
		self.data=[]
		self.get_data()
	def get_data(self):
		class_num=set()
		last=[]
		with open('mnist.txt','r') as f:
			for line in f:
				t=[float(x) for x in line.split(',')]
				self.data.append(t[:-1])
				last.append(t[-1])
		self.feature_num=len(self.data[0])
		self.class_num=len(set(last))
		self.data_num=len(self.data)
		print('class number:',self.class_num)
		self.data=np.array(self.data)
		class_set=sorted(set(last))
		for class_name in class_set:
			print('%i:%i'%(class_name,last.count(class_name)),end='\t')
		#print('1:%i\t-1:%i'%(last.count('1'),last.count('-1')))
		#print(class_num)
	def get_random(self,length,min,max):
		return [random.uniform(min,max) for x in range(length)]

	def k_means(self,data,class_num):
		feature_num=len(data[0])
		the_max=np.max(data)
		the_min=np.min(data)
		classes=[self.get_random(feature_num,the_min,the_max) for x in range(class_num)]
		classes=np.array(classes)
		while 1:
			new=np.zeros((class_num,feature_num))
			count=np.zeros((class_num,1),dtype='int')
			for item in data:
				min=200000000
				min_index=-1
				for index,c in enumerate(classes):
					t=item-c
					dist=np.sum(t*t)
					if dist<min:
						min=dist
						min_index=index
				
				new[min_index]+=item
				count[min_index][0]+=1
			new=new/count
			if (classes==new).all():
				break
			classes=new
			print(count)
	def nmf(self,data):
		data=np.array(data)
		the_max=np.max(data)
		the_min=np.min(data)
		u=[self.get_random(self.class_num,the_min,the_max) for row in range(self.feature_num)]
		v=[self.get_random(self.class_num,the_min,the_max) for row in range(self.data_num)]
		u=np.array(u)
		v=np.array(v)
		x=data.T
		n=50
		while n:
			u=u*np.dot(x,v)/np.dot(np.dot(u,v.T),v)
			v=v*np.dot(x.T,u)/np.dot(np.dot(v,u.T),u)
			n-=1
		u_s=np.sum(u*u,axis=0)**0.5
		t=np.tile(u_s,(self.data_num,1))
		v=v*t
		maxs=np.argmax(v,axis=1)
		count=[0]*self.class_num
		for arg in maxs:
			count[arg]+=1
		print(count)



	def spectral(self,n):
		data=np.array(self.data)
		dists=np.zeros((self.data_num,self.data_num))
		w=np.zeros((self.data_num,self.data_num),dtype='int')
		for row in range(self.data_num):
			for column in range(row,self.data_num):
				t=data[row]-data[column]
				s=np.sum(t*t)
				dists[row][column],dists[column][row]=s,s
		sort_all=np.argsort(dists)
		for row in range(self.data_num):
			for arg in sort_all[row][:n+1]:
				w[row][arg]=-1
				w[arg][row]=-1
		for row in range(self.data_num):
			w[row][row]=-np.sum(w[row])-1
		res=[]
		r,v=np.linalg.eig(w)
		for arg in np.argsort(r)[:self.class_num]:
			res.append(v[arg])
		res=np.dstack(res)[0].tolist()
		self.k_means(res,self.class_num)








if __name__=='__main__':
	c=Clustering()
	#c.k_means(c.data,c.class_num)
	#c.nmf(c.data)
	c.spectral(3)