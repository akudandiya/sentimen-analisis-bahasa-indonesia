from obj.DBM import DBM
import numpy as np
import tensorflow as tf
from mlp import NN
from sklearn.metrics import confusion_matrix, classification_report,  accuracy_score
from sklearn.model_selection import train_test_split
import  gendata

datas = gendata.arrnilais 
labels = gendata.labels
labels = labels.reshape(len(labels),1)

dims = [datas.shape[1],int(datas.shape[1]/2),datas.shape[1]]
learning_rate = 0.01
k1 = 1
k2 = 5
epochs = 10
batch_size = 5
mf = 1
dataset = [tf.cast(tf.reshape(x,shape=(datas.shape[1],1)),"float32") for x in datas]
dbm = DBM(dims, learning_rate, k1, k2, epochs, batch_size)
dbm.train_PCD(dataset)
samples = dbm.block_gibbs_sampling(mean_field = mf, number_samples = datas.shape[0])
sample=[]
for i in samples:
	j=i.numpy().reshape(1,datas.shape[1])
	sample = np.append(sample,j)
samples = sample.reshape(datas.shape[0],datas.shape[1])

X_train, X_test, Y_train, Y_test = train_test_split(samples, labels, test_size=0.2, random_state=0)

n=NN(X_train)
n.train(X_train, Y_train, 3000)
pred = n.think(X_test)
pred = (pred > 0.5).astype(np.int_)

print('confusion_matrix :\n', confusion_matrix(Y_test,pred))
print('classification_report :\n', classification_report(Y_test, pred))
print('accuracy_score :\n', accuracy_score(Y_test, pred))