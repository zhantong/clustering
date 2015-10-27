import random
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
		print('1:%i\t-1:%i'%(last.count('1'),last.count('-1')))
		#print(class_num)

	def get_random(self):
		return [random.random() for x in range(self.feature_num)]
	def test(self):
		classes=[self.get_random() for x in range(self.class_num)]
		#print(c)
		while 1:
			new=[[0]*self.feature_num for x in range(self.class_num)]
			count=[0]*self.class_num
			for item in self.data:
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





if __name__=='__main__':
	c=Clustering()
	c.test()