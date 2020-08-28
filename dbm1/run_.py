from DBM import DBM    
import numpy as np
import joblib
import gendata

def render_output(i,k):
    energy.append(dbm_test.total_energy())
    print( i,'cycle, layer ',k, ' cycle energy: ', energy[-1])
    joblib.dump(dbm_test, 'output/dbm_test')
    joblib.dump(energy, 'output/dbm_energy')
    
def render_supervised(i):
    entropy.append(dbm_test.total_entropy())
    predicts=dbm_test.predict_probs(x_train)
    predict.append(np.mean(np.abs(np.round(predicts)-y_train)))
    accuracy.append(1-np.mean(np.abs(np.round(predicts)-y_train)))
    print( i, ' cycle entropy: ', entropy[-1],' cycle accuracy: ', accuracy[-1])
    joblib.dump(entropy, 'output/dbm_entropy')
    joblib.dump(accuracy, 'output/dbm_accuracy')
    return predicts

x_train = gendata.arrnilais
y_train = gendata.labels
y_train = y_train.reshape(len(y_train),1)

energy = []
entropy = []
accuracy = []
predict = []
predictss=[]
print('initializing model')
dbm_test=DBM(x_train,layers=[30,20])
#render_output(1,1)

for k in range(1,3):
    print ('beginning boltzmann training of model')
    for i in range(10):
        dbm_test.train_unsupervised(k)
        render_output(i,k)

dbm_test.learning_rate = 1.0
dbm_test.add_layer(1)
dbm_test.labels = y_train
#Adapt the output layer to the network
render_output(-1,4)
render_supervised(-1)
for i in range(20):
    #train backprop
    dbm_test.train_backprop(layers=1)
    render_output(i,4)
    render_supervised(i)

#Train the whole thing towards a minimum.
for i in range(20):
    #train backprop
    dbm_test.train_backprop()
    render_output(i,4)
    render_supervised(i)