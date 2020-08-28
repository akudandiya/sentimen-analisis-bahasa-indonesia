from pydbm.dbm.deep_boltzmann_machine import DeepBoltzmannMachine
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm.activation.logistic_function import LogisticFunction
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.relu_function import ReLuFunction
from pydbm.optimization.optparams.sgd import SGD
from sklearn.metrics import confusion_matrix, classification_report,  accuracy_score
from sklearn.model_selection import train_test_split
import gendata
from mlp import NN
import numpy as np

data = gendata.arrnilais #generate idf score from dataset
labels = gendata.labels
labels = labels.reshape(len(labels),1)

# is-a `OptParams`.
opt_params = SGD(
    # Momentum.
    momentum=0.9
)

# Regularization for weights matrix
# to repeat multiplying the weights matrix and `0.9`
# until $\sum_{j=0}^{n}w_{ji}^2 < weight\_limit$.
opt_params.weight_limit = 1e+03

# Probability of dropout.
opt_params.dropout_rate = 0.5

# Contrastive Divergence for visible layer and first hidden layer.
first_cd = ContrastiveDivergence(opt_params=opt_params)
# Contrastive Divergence for first hidden layer and second hidden layer.
second_cd = ContrastiveDivergence(opt_params=opt_params)
activation_list = [
    LogisticFunction(), 
    LogisticFunction(), 
    LogisticFunction()
]

# Setting the object for function approximation.
approximaion_list = [ContrastiveDivergence(), ContrastiveDivergence()]

# DBM
dbm = DeepBoltzmannMachine(
    DBMMultiLayerBuilder(),
    [data.shape[1], 100, data.shape[1]],
    activation_list,
    approximaion_list,
    1e-05 # Setting  rate.
)

# Execute learning.
dbm.learn(
    # `np.ndarray` of observed data points.
    data,
     # If approximation is the Contrastive Divergence, this parameter is `k` in CD method.
    training_count=1,
    # Batch size in mini-batch training.
    batch_size=-1,
    # if `r_batch_size` > 0, the function of `dbm.learn` is a kind of reccursive learning.
    r_batch_size=-1
)

feature_point_arr = dbm.get_feature_point(layer_number=1)
print('\n',feature_point_arr)

X_train, X_test, Y_train, Y_test = train_test_split(feature_point_arr, labels, test_size=0.2, random_state=0)

n=NN(X_train)
n.train(X_train, Y_train, 3000)
pred = n.think(X_test)
pred = (pred > 0.5).astype(np.int_)

print('confusion_matrix :\n', confusion_matrix(Y_test,pred))
print('classification_report :\n', classification_report(Y_test, pred))
print('accuracy_score :\n', accuracy_score(Y_test, pred))