import random
from numpy import *
import time
class Clustering():
	def __init__(self):
		self.data=[]
		self.get_data()
	def get_data(self):
		class_num=set()
		last=[]
		with open('german.txt','r') as f:
			for line in f:
				t=[float(x) for x in line.split(',')[:-1]]
				self.data.append(t)
				class_num.add(line.rsplit(',',1)[-1].strip())
				last.append(line.rsplit(',',1)[-1].strip())
		self.feature_num=len(self.data[0])
		self.class_num=len(class_num)
		self.data_num=len(self.data)
		print('1:%i\t-1:%i'%(last.count('1'),last.count('-1')))
		#print(class_num)

	def get_random(self):
		#return [random.random() for x in range(self.feature_num)]
		return [random.uniform(-0.2,0.5) for x in range(self.feature_num)]
	def k_means(self,data,class_num):
		classes=[self.get_random() for x in range(class_num)]
		feature_num=len(data[0])
		#print(c)
		while 1:
			new=[[0]*feature_num for x in range(class_num)]
			count=[0]*class_num
			for item in data:
				min=200
				min_index=-1
				for index,c in enumerate(classes):
					dist=sum([(a-b)*(a-b) for (a,b) in zip(item,c)])
					if dist<min:
						min=dist
						min_index=index
				
				new[min_index]=[a+b for (a,b) in zip(new[min_index],item)]
				count[min_index]+=1
			#print(new)
			classes=[]
			for index,c in enumerate(new):
				classes.append([item/count[index] for item in c])
			print(count)
			print(classes)
			#break
	def nmf(self):
		u=[[random.random() for column in range(self.class_num)] for row in range(self.feature_num)]
		v=[[random.random() for column in range(self.class_num)] for row in range(self.data_num)]
		u=array(u)
		v=array(v)
		x=array(self.data).T
		n=50
		while n:
			#print(len(v),len(v[0]))
			u=u*dot(x,v)/dot(dot(u,v.T),v)
			v=v*dot(x.T,u)/dot(dot(v,u.T),u)
			n-=1
		u=u.tolist()
		v=v.tolist()
		#print(len(v),len(v[0]))
		for column in range(self.class_num):
			s=0
			for u_row in range(self.feature_num):
				s+=u[u_row][column]*u[u_row][column]
			s=s**0.5
			for row in range(self.data_num):
				v[row][column]*=s
		#print(v[:10])
		count=[0]*self.class_num
		for f in v:
			count[f.index(max(f))]+=1
		print(count)

	def spectral_bak(self,n):
		t1=time.time()
		w=[[0]*self.data_num for x in range(self.data_num)]
		connect={}
		for row in range(self.data_num):
			dists={}
			for column in range(self.data_num):
				dists[column]=sum([(a-b)*(a-b) for (a,b) in zip(self.data[row],self.data[column])])
			for column in sorted(dists,key=lambda k:dists[k])[1:n+1]:
					w[row][column],w[column][row]=-1,-1
		t2=time.time()
		print((t2-t1))
		for row in range(len(w)):
			s=-sum(w[row])
			w[row][row]=s
		w=array(w)
		r,v=linalg.eig(w)
		#for i in range(4):
			#print(w[i])
		res=[]
		r=r.tolist()
		#v=v.tolist()
		for item in sorted(r)[:self.class_num]:
			res.append(v[r.index(item)])
		res=dstack(res)[0].tolist()
		print(len(res),len(res[0]))
		print(res[:10])
		t3=time.time()
		print(t3-t2)
		self.k_means(res,self.class_num)
		#print(res)

	def spectral(self,n):
		dists=[[0]*self.data_num for x in range(self.data_num)]
		w=[[0]*self.data_num for x in range(self.data_num)]
		d=[[0]*self.data_num for x in range(self.data_num)]
		w=array(w)
		d=array(d)
		for row in range(self.data_num):
			for column in range(row,self.data_num):
				s=0
				for (a,b) in zip(self.data[row],self.data[column]):
					s+=pow((a-b),2)
				dists[row][column],dists[column][row]=s,s
		for row in range(self.data_num):
			for item in sorted(dists[row])[1:n+1]:
				w[row][dists[row].index(item)]=1
				w[dists[row].index(item)][row]=1
		for row in range(self.data_num):
			d[row][row]=sum(w[row])
		l=d-w
		r,v=linalg.eig(l)
		r=r.tolist()
		res=[]
		for item in sorted(r)[:self.class_num]:
			res.append(v[r.index(item)])
		res=dstack(res)[0].tolist()
		the_max=0
		the_min=0
		for row in range(len(res)):
			c_max=max(res[row])
			c_min=min(res[row])
			if c_max>the_max:
				the_max=c_max
			if c_min<the_min:
				the_min=c_min
		print(the_max,the_min)
		#print(sorted(r)[:20])
		self.k_means(res,self.class_num)








if __name__=='__main__':
	c=Clustering()
	#c.k_means(c.data,c.class_num)
	c.spectral(3)