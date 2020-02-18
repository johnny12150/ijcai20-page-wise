import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import ml_metrics as metrics
from model.baseline.graphsage_dnn.Layers import Dense

with open('F:/volume/0217graphsage/0106/all_edge.pkl', 'rb') as file:
    all_edge = pickle.load(file)

# filter out relation = 0 (reference)
all_paper = all_edge.loc[all_edge.rel == 0]
all_paprr_id = list(set(all_paper['head'].tolist() + all_paper['tail'].tolist()))

# load paper embedding
paper_emb = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/embedding.npy')
emb_node = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/emb_node.npy')
# keep node and emb only in paper_id
index = np.where(np.in1d(emb_node, all_paprr_id))
emb_node = emb_node[np.in1d(emb_node, index)]
paper_emb = paper_emb[index]

# use trained tf NN predict cite or not
tf.reset_default_graph()
saver = tf.train.import_meta_graph('F:/volume/0217graphsage/0106/model_output/model.meta')
# https://stackoverflow.com/questions/44251328/tensorflow-print-all-placeholder-variable-names-from-meta-graph
imported_graph = tf.get_default_graph()
graph_op = imported_graph.get_operations()
# print([x for x in tf.get_default_graph().get_operations() if x.type == "Placeholder"])

# with open('./model/baseline/NN_weights.txt', 'w') as f:
#     for i in graph_op:
#         f.write(str(i))

graph_b0 = tf.get_default_graph().get_operation_by_name('dense_1_vars/bias')
graph_b1 = tf.get_default_graph().get_operation_by_name('dense_1_vars/bias_1')
print(graph_b0.values())

# https://stackoverflow.com/questions/42769435/get-variable-does-not-work-after-session-restoration
sess = tf.Session()
saver.restore(sess, 'F:/volume/0217graphsage/0106/model_output/model')
# saver.restore(sess, tf.train.latest_checkpoint('F:/volume/0217graphsage/0106/model_output'))

# FIXME error: You must feed a value for placeholder tensor 'Placeholder' with dtype
sess.run(tf.global_variables_initializer())
# print(tf.get_variable("bias"))
print(sess.run('dense_1_vars/bias'))

# get weight, https://ithelp.ithome.com.tw/articles/10187786
all_vars = tf.trainable_variables()
# print(all_vars[6].name)
# print(sess.run(all_vars[6]).shape)
# call layer with saved weights
node_pred = Dense(all_vars[5], all_vars[6], all_vars[7], all_vars[8], act=lambda x: x)

sample = np.zeros((2, 100)).astype('float32')
node_preds = node_pred(sample)
print(node_preds.eval(session=sess))

# print(np.where(emb_node == 1))  # where is the paper id
# print(paper_emb[np.where(emb_node == 1)[0]][0].shape)  # get emb by paper id


# create paper pair
def gen_paper(nodes, batch_i):
    n_times = nodes.shape[0]
    index_i = np.where(emb_node == batch_i)[0]
    paper_i_emb = paper_emb[index_i][0]
    exclude_i = np.delete(paper_emb, index_i, axis=0)  # exclude row i
    # fixme times可能不用減 1
    paper_i_pair = np.array([paper_i_emb.tolist(), ] * (n_times-1))  # repeat rows
    # paper_i_pair = np.repeat(paper_i_pair, (n_times-1))
    paper_i_pair = np.concatenate((paper_i_pair, exclude_i), axis=1)
    batch_y = all_paper[all_edge['head'] == batch_i]['tail'].values  # find tail ids
    # set 1 at tail index
    # i_cited_index = np.where(np.in1d(emb_node, all_paper[all_edge['head'] == 0]['tail'].values))
    # batch_y = np.zeros(n_times-1)
    # batch_y[np.in1d(batch_y, i_cited_index)] = 1
    batch_x = paper_i_pair
    yield batch_x, batch_y, np.delete(emb_node, index_i)


# fixme should use all_paper_id, instead of emb_node
# to reduce memory usage, use batch prediction
# batch_paths = np.random.choice(emb_node, size=emb_node.shape[0])
batch_paths = np.random.choice(emb_node, size=100)
ans = []
rs = []
K = 150
i = 0
for batch_i in tqdm(batch_paths):
    x, y, classes = next(gen_paper(emb_node, batch_i))
    # if output value > 0.5, keep classes
    i_prediction = sess.run(node_pred(x.astype('float32'))).reshape(-1)
    # sort classes and output at the same time
    sorter = i_prediction.argsort()[::-1]  # reverse
    i_prediction = i_prediction[sorter][:K]
    classes = classes[sorter][:K]
    ans.append(y.tolist())
    rs.append(classes.tolist())

# todo calculate MAP
print(metrics.mapk(ans, rs, K))

