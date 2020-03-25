import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import ml_metrics as metrics
from model.baseline.graphsage_dnn.Layers import custom_Dense, zeros
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

task = 0
with open('F:/volume/0217graphsage/0106/all_edge.pkl', 'rb') as file:
    all_edge = pickle.load(file)

# 把df原始 id map到 graphsage node的 id
# with open('F:/volume/0217graphsage/0106/id_map.pkl', 'rb') as file:
#     id_map = pickle.load(file)
# all_edge['head'] = all_edge['head'].map(id_map)
# all_edge['tail'] = all_edge['tail'].map(id_map)

# filter out relation = 0 (reference)
all_paper = all_edge.loc[all_edge.rel == 0]
all_paper_id = list(set(all_paper['head'].tolist() + all_paper['tail'].tolist()))
candidate_ids = list(set(all_paper['tail'].tolist()))
# 只有出現在head過的paper
head_paper_ids = list(set(all_paper['head'].tolist()))

# load paper embedding
paper_emb = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/embedding.npy')
emb_node = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/emb_node.npy')
# keep node and emb only in paper_id
index = np.where(np.in1d(emb_node, all_paper_id))
emb_node = emb_node[index]
paper_emb = paper_emb[index].astype('float32')
# find candidate papers' embedding
c_index = np.where(np.in1d(emb_node, candidate_ids))
candidate_paper_emb = paper_emb[c_index]


# create paper pair
def gen_paper(batch_i, pred=True):
    index_i = np.where(np.in1d(emb_node, batch_i))[0]
    paper_i_emb = paper_emb[index_i]
    # exclude_i = np.delete(paper_emb, index_i, axis=0)  # exclude row i
    # paper_i_pair = np.array([paper_i_emb.tolist(), ] * (n_times-1))  # repeat rows
    paper_i_pair = np.tile(paper_i_emb, (len(candidate_ids))).reshape(-1, 100)  # repeat rows
    candidates = np.tile(candidate_paper_emb, (len(batch_i))).reshape(-1, 100)  # repeat candidate_ids
    paper_i_pair = np.concatenate((paper_i_pair, candidates), axis=1)  # shape: N * len(candidate_ids) * 200
    batch_y = []
    for j in batch_i:
        batch_y.append(all_paper[all_paper['head'] == j]['tail'].tolist())  # find tail ids

    if pred:
        # prediction
        classes = make_prediction(paper_i_pair, len(batch_i))  # shape: len(x) * K
        yield classes, batch_y
    else:
        yield paper_i_pair, batch_y


def load_paper_pair(path):
    paper_size = 60
    end = paper_size * len(candidate_ids)
    batch_x = np.load(path[0], allow_pickle=True)[:end]
    batch_ans = np.load(path[1], allow_pickle=True)[:end].tolist()
    batch_pred = make_prediction(batch_x, paper_size).tolist()
    print(metrics.mapk(batch_ans, batch_pred, 150))


# use trained tf NN to predict cite or not
def make_prediction(x, size):
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
            # use tensor_board to check nodes
            # writer = tf.summary.FileWriter("TensorBoard/", graph=sess.graph)
            # print(x.nbytes / 10 ** 9)
            i_prediction = sess.run(tf.nn.sigmoid(node_pred(x))).reshape(size, -1)
            # print(len([n.name for n in tf.get_default_graph().as_graph_def().node]))  # check tf graph size

        # sort classes and output at the same time
        # https://stackoverflow.com/questions/33140674/argsort-for-a-multidimensional-ndarray
        # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
        new_sorter = i_prediction.argsort(axis=1)[:, -K:]  # sort and select first 150
        new_sorter = np.flip(new_sorter, axis=1)
        batch_classes = np.tile(candidate_ids, (size, 1))  # shape: N * len(candidate_ids)
        classes = np.take_along_axis(batch_classes, new_sorter, axis=1)
    return classes


if task == 0:
    # to reduce memory usage, use batch prediction
    # batch_paths = np.random.choice(head_paper_ids, size=300)
    ans = []
    rs = []
    K = 150
    batch_size = 20

    for i in tqdm(range(50)):
        batch_paths = head_paper_ids[i*batch_size:(i+1)*batch_size]  # 一次處理20筆
        pred, y_label = next(gen_paper(batch_paths))
        ans.extend(y_label)
        rs.extend(pred.tolist())

    # calculate MAP
    print(metrics.mapk(ans, rs, K))

    # for j in tqdm(range(20)):
        # 先把所有 pair分批產好並存到disk, 之後再做 predict
        # np.save('npy_temp/batch'+str(batch_size)+'_x_emb.npy', np.array(batch_x))
        # np.save('npy_temp/batch'+str(batch_size)+'_y_label.npy', np.array(batch_y))

    # test這300篇的MAP
    # paths = ['npy_temp/batch300_x_emb.npy', 'npy_temp/batch300_y_label.npy']
    # load_paper_pair(paths)


# check graphsage dnn
def sage_nn(nodes, save=True):
    x = []
    y = []
    for batch_i in tqdm(nodes):
        # 找node_i真實有site的, 讓nn判斷
        index_i = np.where(emb_node == batch_i)[0]
        paper_i_emb = paper_emb[index_i]  # paper i embedding
        cite_paper = all_paper[all_paper['head'] == batch_i]['tail'].values
        i_cited_index = np.where(np.in1d(emb_node, cite_paper))
        paper_i_cite_emb = paper_emb[i_cited_index]  # paper i 有cite的embedding
        num_cited = paper_i_cite_emb.shape[0]  # paper i cite 幾篇
        # add negative sample
        num_neg_sample = 10  # 設定negative sample數量
        # neg_index = np.where(~np.in1d(candidate_ids, cite_paper))[0]

        # negative index就是positive的差集
        # neg_index = list(set(range(len(emb_node))) - set(i_cited_index[0].tolist()))
        # neg_index = np.random.choice(neg_index, num_neg_sample)  # random select

        neg_index = np.where(np.in1d(emb_node, np.random.choice(candidate_ids, size=num_neg_sample)))[0]
        neg_sample_emb = paper_emb[neg_index]
        num_neg_sample = neg_sample_emb.shape[0]  # avoid node not exist
        if num_cited > 0:
            # repeat_emb = np.tile(paper_i_emb, (num_cited + num_neg_sample)).reshape(-1, 100)  # clone rows
            repeat_emb = np.tile(paper_i_emb, (num_cited + num_neg_sample, 1))
            embs = np.concatenate((paper_i_cite_emb, neg_sample_emb), axis=0)
            x.extend(np.concatenate((repeat_emb, embs), axis=1))  # positive sample
            # x.extend(np.concatenate((repeat_emb, paper_i_cite_emb), axis=1))  # positive sample
            # x.extend(np.concatenate((np.tile(paper_i_emb, num_neg_sample).reshape(-1, 100), neg_sample_emb), axis=1))  # add negative sample
            y.extend([1] * num_cited + [0] * num_neg_sample)

    if save:
        np.save('./paper_pair.npy', np.array(x))
        np.save('./pair_label.npy', np.array(y))
    return np.array(x), np.array(y)


if task == 1:
    tf.reset_default_graph()
    h1_dim, h2_dim = 300, 100
    w0 = tf.get_variable('dense_1_vars/weights', shape=(200, h1_dim))  # define variables that we want
    w1 = tf.get_variable('dense_1_vars/weights_1', shape=(h1_dim, h2_dim))
    w2 = tf.get_variable('dense_1_vars/weights_2', shape=(h2_dim, 1))
    b0 = zeros(h1_dim, 'dense_1_vars/bias')
    b1 = zeros(h2_dim, 'dense_1_vars/bias_1')
    b2 = zeros(1, 'dense_1_vars/bias_2')
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, 'F:/volume/0217graphsage/0106/model_output/model')
    all_vars = tf.trainable_variables()
    # call layer with saved weights
    node_pred = custom_Dense(w0, w1, w2, b0, b1, b2)
    x, y = sage_nn(head_paper_ids)
    # shuffle x, y
    # p = np.random.permutation(len(x))
    # x, y = x[p], y[p]
    y_logit = y.astype('float32').reshape(-1, 1)
    prediction_logit = sess.run(node_pred(x.astype('float32')))  # predict
    loss = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_logit, labels=y_logit))
    print(np.sum(loss.reshape(-1))/ len(loss))
    prediction = sess.run(tf.nn.sigmoid(prediction_logit))
    prediction = np.round(prediction.reshape(-1))  # flatten & to 0/ 1
    print(np.sum(np.equal(prediction, y)) / len(y))
    sess.close()

