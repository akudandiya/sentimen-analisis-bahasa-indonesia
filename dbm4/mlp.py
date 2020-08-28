import numpy as np

def sigmoid(x):
		return 1.0/(1.0 + np.exp(-x)) 

def sigmoid_der(x):
		return x*(1.0 - x) 
		
class NN: 
	def __init__(self, inputs): 
		self.inputs = inputs 
		self.lc=self.inputs.shape[0] 
		self.lr=self.inputs.shape[1] 
		self.wi=np.random.random((self.lr, self.lc)) 
		print("wi:",self.wi.shape) 
		self.wh=np.random.random((self.lc, 1)) 
		print("wh:",self.wh.shape)
		
	def think(self, inp): 
		s1=sigmoid(np.dot(inp, self.wi)) 
		s2=sigmoid(np.dot(s1, self.wh)) 
		return s2
		
	def train(self, inputs,outputs, it): 
		for i in range(it): 
			l0=inputs 
			l1=sigmoid(np.dot(l0, self.wi)) 
			l2=sigmoid(np.dot(l1, self.wh)) #preds
			
			l2_err=l2*(1-l2)*(outputs-l2) #y - preds
			l2_delta = np.multiply(0.1,np.multiply(l2_err, sigmoid_der(l2)))
			
			l1_err=np.dot(l2_delta, self.wh.T)
			l1_delta=np.multiply(l1_err, sigmoid_der(l1))
			
			self.wh+=np.dot(l1.T, l2_delta)
			self.wi+=np.dot(l0.T, l1_delta)