import numpy as np
from mlp import NN
import gendata
from pydbm.dbm.deepboltzmannmachine.stacked_auto_encoder import StackedAutoEncoder
from pydbm.dbm.builders.dbm_multi_layer_builder import DBMMultiLayerBuilder
from pydbm.approximation.contrastive_divergence import ContrastiveDivergence
from pydbm.activation.tanh_function import TanhFunction
from pydbm.activation.logistic_function import LogisticFunction
from sklearn.metrics import confusion_matrix, classification_report,  accuracy_score
from sklearn.model_selection import train_test_split

data = gendata.arrnilais #generate idf score from dataset
labels = gendata.labels
#target_arr = np.random.normal(loc=0.0, scale=1.0, size=(100, 50))
#target_arr = (target_arr - target_arr.mean()) / target_arr.std()
labels = labels.reshape(len(labels),1)

activation_list = [LogisticFunction(), LogisticFunction(), LogisticFunction()]

approximaion_list = [ContrastiveDivergence(), ContrastiveDivergence()]

dbm = StackedAutoEncoder(
    DBMMultiLayerBuilder(),
    [data.shape[1], 100, data.shape[1]],
    activation_list,
    approximaion_list,
    1e-03 # Setting  rate.
)

dbm.learn(
    data,
    training_count=10, # If approximation is the Contrastive Divergence, this parameter is `k` in CD method.
    batch_size=-1,  # Batch size in mini-batch training.
    r_batch_size=-1,  # if `r_batch_size` > 0, the function of `dbm.learn` is a kind of reccursive learning.
    sgd_flag=True
)

# layer_number corresponds to the index of approximaion_list. 
# And reconstruct_error_arr is the np.ndarray of reconstruction error rates.
reconstruct_error_arr = dbm.get_reconstruct_error_arr(layer_number=0)

pre_trained_arr = dbm.feature_points_arr

weight_arr_list = dbm.get_weight_arr_list()

visible_bias_arr_list = dbm.get_visible_bias_arr_list()
hidden_bias_arr_list = dbm.get_hidden_bias_arr_list()

dbm.save_pre_learned_params(dir_path="pre-learned", file_name="dbm_demo")

dbm_t = StackedAutoEncoder(
    DBMMultiLayerBuilder(
        pre_learned_path_list=[
            "pre-learned/dbm_demo_0.npz",
            "pre-learned/dbm_demo_1.npz"
        ]
    ),
    [data.shape[1], 100, data.shape[1]],
    activation_list,
    approximaion_list,
    1e-03#Learning rate.
)

# Execute learning.
dbm_t.learn(
    data,
    training_count=10, # If approximation is the Contrastive Divergence, this parameter is `k` in CD method.
    batch_size=-1,  # Batch size in mini-batch training.
    r_batch_size=-1  # if `r_batch_size` > 0, the function of `dbm.learn` is a kind of reccursive learning.
)
trained_arr = dbm_t.feature_points_arr
reconstruct_error_arr = dbm_t.get_reconstruct_error_arr(layer_number=0)

X_train, X_test, Y_train, Y_test = train_test_split(trained_arr, labels, test_size=0.2, random_state=0)

n=NN(X_train)
n.train(X_train, Y_train, 3000)
pred = n.think(X_test)
pred = (pred > 0.5).astype(np.int_)

print('confusion_matrix :\n', confusion_matrix(Y_test, pred))
print('classification_report :\n', classification_report(Y_test, pred))
print('accuracy_score :\n', accuracy_score(Y_test, pred))