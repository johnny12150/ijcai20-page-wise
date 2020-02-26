# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)


# %%
import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
#import ml_metrics as metrics
#from model.baseline.graphsage_dnn.Layers import Dense


# %%
#f = open('dict_paper_key.pkl','rb')
#paper_dict = pickle.load(f)
#paper_dict[50871]


# %%
#import pandas as pd
#all_edge = pd.read_pickle('../all_edge.pkl')
#all_paper = all_edge.loc[all_edge.rel == 0]
#all_paprr_id = list(set(all_paper['head'].tolist() + all_paper['tail'].tolist()))
#candidate_ids = list(set(all_paper['tail'].tolist()))    


# %%
with open('F:/volume/0217graphsage/0106/all_edge.pkl', 'rb') as file:
    all_edge = pickle.load(file)
all_paper = all_edge.loc[all_edge.rel == 0]
paper_emb = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/embedding.npy')
emb_node = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/emb_node.npy')

# fixme keep node and emb only in paper_id
all_paper_id = list(set(all_paper['head'].tolist() + all_paper['tail'].tolist()))
candidate_ids = list(set(all_paper['tail'].tolist()))
index = np.where(np.in1d(emb_node, all_paper_id))
emb_node = emb_node[index]
paper_emb = paper_emb[index]

# %%
tf.reset_default_graph()
saver = tf.train.import_meta_graph('F:/volume/0217graphsage/0106/model_output/model.meta')
imported_graph = tf.get_default_graph()
graph_op = imported_graph.get_operations()


# %%
sess = tf.Session()
saver.restore(sess,  'F:/volume/0217graphsage/0106/model_output/model')


# %%
all_vars = tf.trainable_variables()


# %%
import tensorflow as tf


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).
    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', True)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            #            if self.logging and not self.sparse_inputs:
            #                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class Dense(Layer):
    """Dense layer."""

    def __init__(self, w0, w1, w2, b0, b1, b2, dropout=0.,
                 act=tf.nn.relu, placeholders=None, bias=True, featureless=False,
                 sparse_inputs=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.sparse_inputs = sparse_inputs
        if sparse_inputs:
            self.num_features_nonzero = placeholders['num_features_nonzero']
        '''用tf.variable_scope才可以重複使用變量'''
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = w0
            self.vars['weights_1'] = w1
            self.vars['weights_2'] = w2
            self.vars['bias'] = b0
            self.vars['bias_1'] = b1
            self.vars['bias_2'] = b2

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs
        # transform
        hidden_1 = tf.matmul(x, self.vars['weights'])
        hidden_1 += self.vars['bias']
        hidden_1 = self.act(hidden_1)
        hidden_2 = tf.matmul(hidden_1, self.vars['weights_1'])
        hidden_2 += self.vars['bias_1']
        hidden_2 = self.act(hidden_2)
        hidden_3 = tf.matmul(hidden_2, self.vars['weights_2'])
        hidden_3 += self.vars['bias_2']
#        output = tf.nn.sigmoid(hidden_3)
        output = hidden_3
        return output


# %%
node_pred = Dense(all_vars[5], all_vars[6], all_vars[7], all_vars[8], all_vars[9], all_vars[10])


# %%
temp = []
label = []
for i in range(0, 200000):
    first_paper = np.where(emb_node == all_paper['head'].iloc[i])[0]
    second_paper = np.where(emb_node == all_paper['tail'].iloc[i])[0]
    x = np.concatenate((paper_emb[first_paper], paper_emb[second_paper]),axis=1)
#    print(x.shape)
#    x_input = np.reshape(x,(1,200))
    temp.append(x)
    label.append(1)
    # first_negative_paper = np.where(emb_node == all_paper['head'].iloc[i])[0]
    for j in range(5):
        # second_negative_paper = np.where(emb_node == np.random.randint(60000, size=1))[0]
        second_negative_paper = np.where(np.in1d(emb_node, np.random.choice(candidate_ids, size=1)))[0]
        x = np.concatenate((paper_emb[first_paper], paper_emb[second_negative_paper]),axis=1)
        # x = np.concatenate((np.tile(paper_emb[first_paper], 2).reshape(-1, 100), paper_emb[second_negative_paper]), axis=1)
    #    x_input = np.reshape(x,(1,len(x)))
        temp.append(x)
        label.append(0)


# %%
label = np.array(label).astype('float32')
label = np.reshape(label,(len(label),1))

# %%
temp = np.reshape(temp,(len(temp),200))
temp = np.array(temp).astype('float32')

# %%
node_preds = node_pred((temp.astype('float32')))
print(node_preds.eval(session=sess))
loss = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(logits=node_preds, labels=label))
print('loss : ' + str(np.sum(loss.reshape(-1))/ len(loss)))
prediction = sess.run(tf.nn.sigmoid(node_preds))
prediction = np.round(prediction)  # flatten & to 0/ 1
print('acc : ' + str(np.sum(np.equal(prediction, label))/ len(label)))

