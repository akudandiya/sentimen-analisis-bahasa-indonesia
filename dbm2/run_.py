import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,  accuracy_score
from dbm import DBM
from mlp import NN
import gendata

data = gendata.arrnilais
labels = gendata.labels
labels = labels.reshape(len(labels),1)

dbm = DBM(n_visible=data.shape[1], n_hidden1=100, n_hidden2=data.shape[0],
          n_chains=100, n_vi_steps=10, n_gibbs_steps=1,
          learning_rate=0.5, early_stopping=False,
          seed=21)
          
n_epoch = 5
dbm.train(data, 
          batch_size=2, n_epoch=n_epoch, batch_seed=24, 
          verbose=True)
          
#dbm.save('models/dbm-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(
#         dbm.n_hidden1, dbm.n_hidden2, dbm.n_chains, 
#         dbm.n_vi_steps, dbm.n_gibbs_steps, 
#         dbm.learning_rate.const, dbm.early_stopping,
#         dbm.epoch))
#         

X_train, X_test, Y_train, Y_test = train_test_split(dbm.W2, labels, test_size=0.2, random_state=0)

n = NN(X_train)
n.train(X_train, Y_train, 3000)
pred = n.think(X_test)
pred = (pred > 0.5).astype(np.int_)

print('confusion_matrix :\n', confusion_matrix(Y_test, pred))
print('classification_report :\n', classification_report(Y_test, pred))
print('accuracy_score :\n', accuracy_score(Y_test, pred))