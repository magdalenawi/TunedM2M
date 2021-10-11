from data_load import *
from TunedM2M import *
from utils import *
n = 20
lr = 3e-3
layer = 64
layer_num = 15
dropout = 0.7
dropout_att = 0.3
reg = 3e-3
heads = 8
beta = 1.0

def train_test():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('solubility')
    N, D = features.shape
    print('feature shape: ', (N, D))
    features = preprocess_features(features)
    print('feature preproecess ended')
    adj = markov(adj)
    print('adj preproecess ended')
    print('y_train shape: ', y_train.shape)
    print('test_set: ', np.sum(test_mask))
    print('train_set: ',np.sum(train_mask))
    print('dataset loading ended..')

    tf.reset_default_graph()
    model = TunedM2M(layer, layer_num, heads, N, D, y_train.shape[1], lr, reg,
                 prob=dropout, prob_att=dropout_att, beta=beta, 
                 adj_idx=adj[0], idx_size=int(len(adj[0])*0.05))

    print('start training..')
    min_val_loss = 100
    max_val_acc = 0
    val_acc_save = []
    val_loss_save = []
    for epoch in range(300):
        model.train(features, y_train, train_mask, adj)
        val_loss = model.get_loss(features, y_val, val_mask, adj)
        val_acc = model.accuracy(features, y_val, val_mask, adj)
        val_acc_save.append(val_acc)
        val_loss_save.append(val_loss)
        """
        Early stopping...
        """
        if val_acc >= max_val_acc or val_loss <= min_val_loss:
            if val_acc >= max_val_acc and val_loss <= min_val_loss:
                test_acc = model.accuracy(features, y_test, test_mask, adj)
            max_val_acc = np.max(val_acc_save)
            min_val_loss = np.min(val_loss_save)
            step_counter = 0
        else:
            step_counter += 1
            if step_counter == n:
                break
        """
        Early stopping ended...
        """
    print(test_acc * 100)
