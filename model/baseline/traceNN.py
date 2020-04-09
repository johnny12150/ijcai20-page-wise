import pickle
import tensorflow as tf
import numpy as np
import ml_metrics as metrics
from model.baseline.graphsage_dnn.Layers import custom_Dense, zeros
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with open('F:/volume/0217graphsage/0106/all_edge_1y.pkl', 'rb') as file:
    all_edge = pickle.load(file)

# filter out relation = 0 (reference)
all_paper = all_edge.loc[all_edge.rel == 0]
# np in1d沒照 list的順序
all_paper_id = np.sort(np.array(list(set(all_paper['head'].tolist() + all_paper['tail'].tolist()))))
candidate_ids = np.sort(np.array(list(set(all_paper['tail'].tolist()))))
# 只有出現在head過的paper
head_paper_ids = list(set(all_paper['head'].tolist()))

# load paper embedding
paper_emb = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/embedding.npy')
emb_node = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/emb_node.npy')
sort_ = np.argsort(emb_node)
emb_node = emb_node[sort_]
paper_emb = paper_emb[sort_]
# keep node and emb only in paper_id
index = np.where(np.in1d(emb_node, all_paper_id))
emb_node = emb_node[index]
paper_emb = paper_emb[index].astype('float32')
# find candidate papers' embedding
c_index = np.where(np.in1d(emb_node, candidate_ids))
candidate_paper_emb = paper_emb[c_index]

# 找最多ref的paper
all_ans = all_paper.astype(str).groupby('head')['tail'].agg([','.join])
print(all_ans.index.values.astype(int).min())
print(all_paper.groupby('head')['tail'].count().reset_index(name='count').sort_values(['count'], ascending=False))


def gen_paper(batch_i):
    index_i = np.where(np.in1d(emb_node, batch_i))[0]
    paper_i_emb = paper_emb[index_i]
    paper_i_pair = np.tile(paper_i_emb, (len(candidate_ids))).reshape(-1, 100)  # repeat rows
    candidates = np.tile(candidate_paper_emb, (1, 1))  # repeat candidate_ids
    paper_i_pair = np.concatenate((paper_i_pair, candidates), axis=1)  # shape: N * len(candidate_ids) * 200
    batch_y = []
    batch_y.append(all_paper[all_paper['head'] == batch_i]['tail'].tolist())  # find tail ids

    yield paper_i_pair, batch_y


batch_paths = head_paper_ids[0]  # id: 21387有81個ref
x, y = next(gen_paper(21387))
size = 1
h1_dim, h2_dim = 300, 100

with tf.Graph().as_default():
    w0 = tf.get_variable('dense_1_vars/weights', shape=(200, h1_dim))  # define variables that we want
    w1 = tf.get_variable('dense_1_vars/weights_1', shape=(h1_dim, h2_dim))
    w2 = tf.get_variable('dense_1_vars/weights_2', shape=(h2_dim, 1))
    b0 = zeros(h1_dim, 'dense_1_vars/bias')
    b1 = zeros(h2_dim, 'dense_1_vars/bias_1')
    b2 = zeros(1, 'dense_1_vars/bias_2')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, 'F:/volume/0217graphsage/0106/model_output/model')
        node_pred = custom_Dense(w0, w1, w2, b0, b1, b2)
        i_prediction = sess.run(tf.nn.sigmoid(node_pred(x))).reshape(size, -1)

    # 研究 prediction
    print(len(np.where(i_prediction > 0.5)[0]))  # 會有315個大於0.5, 157大於0.9
    # find where is the ans index
    ans_where = np.where(np.in1d(candidate_ids, y))  # 答案的 emb index
    # print(i_prediction.reshape(-1)[ans_where])  # 找答案的預估值
    print(len(np.where(i_prediction.reshape(-1)[ans_where] > 0.5)[0]))  # NN猜的答案有幾個會大於0.5

    new_sorter = i_prediction.argsort(axis=1)[:, -150:]  # sort and select first 150
    new_sorter = np.flip(new_sorter, axis=1)
    batch_classes = np.tile(candidate_ids, (size, 1))  # shape: N * len(candidate_ids)
    classes = np.take_along_axis(batch_classes, new_sorter, axis=1)
    # 算 hit
    print(np.where(np.in1d(classes.reshape(-1), y))[0].shape)
