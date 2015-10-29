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
	def get_random(self,length,min,max):
		#return [random.random() for x in range(self.feature_num)]
		return [random.uniform(min,max) for x in range(length)]
	def k_means_bak(self,data,class_num):
		feature_num=len(data[0])
		the_max=np.max(data)
		the_min=np.min(data)
		classes=[self.get_random(feature_num,the_min,the_max) for x in range(class_num)]
		#classes=np.array(classes)
		#print(c)
		while 1:
			new=[[0]*feature_num for x in range(class_num)]
			#new=np.array(new)
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
			#print(classes)
			#break
	def k_means(self,data,class_num):
		feature_num=len(data[0])
		the_max=np.max(data)
		the_min=np.min(data)
		classes=[self.get_random(feature_num,the_min,the_max) for x in range(class_num)]
		classes=np.array(classes)
		#print(c)
		while 1:
			#new=[[0]*feature_num for x in range(class_num)]
			new=np.zeros((class_num,feature_num))
			#count=[0]*class_num
			count=np.zeros((class_num,1),dtype='int')
			for item in data:
				min=200
				min_index=-1
				for index,c in enumerate(classes):
					t=item-c
					dist=np.sum(t*t)
					#dist=sum([(a-b)*(a-b) for (a,b) in zip(item,c)])
					if dist<min:
						min=dist
						min_index=index
				
				new[min_index]+=item
				count[min_index][0]+=1
			#print(new)
			new=new/count
			if (classes==new).all():
				break
			classes=new
			print(count)
			#print(classes)
			#break
	def nmf(self,data):
		data=np.array(data)
		u=[[random.random() for column in range(self.class_num)] for row in range(self.feature_num)]
		v=[[random.random() for column in range(self.class_num)] for row in range(self.data_num)]
		u=np.array(u)
		v=np.array(v)
		x=data.T
		n=50
		while n:
			#print(len(v),len(v[0]))
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

#		u=u.tolist()
#		v=v.tolist()
#		#print(len(v),len(v[0]))
#		for column in range(self.class_num):
#			s=0
#			for u_row in range(self.feature_num):
#				s+=u[u_row][column]*u[u_row][column]
#			s=s**0.5
#			for row in range(self.data_num):
#				v[row][column]*=s
#		#print(v[:10])
#		count=[0]*self.class_num
#		for f in v:
#			count[f.index(max(f))]+=1
#		print(count)

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
	#c.k_means(np.array(c.data),c.class_num)
	c.nmf(c.data)
	#c.spectral(3)