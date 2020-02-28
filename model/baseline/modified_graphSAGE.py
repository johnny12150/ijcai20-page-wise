import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import ml_metrics as metrics
from model.baseline.graphsage_dnn.Layers import custom_Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

task = 0
with open('F:/volume/0217graphsage/0106/all_edge.pkl', 'rb') as file:
    all_edge = pickle.load(file)

# 把df原始 id map到 graphsage node的 id
# with open('F:/volume/0217graphsage/0106/id_map.pkl', 'rb') as file:
#     id_map = pickle.load(file)

# 有的化, loss = 2.667, acc = 0.51
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

# use trained tf NN to predict cite or not
tf.reset_default_graph()
saver = tf.train.import_meta_graph('F:/volume/0217graphsage/0106/model_output/model.meta')
# https://stackoverflow.com/questions/44251328/tensorflow-print-all-placeholder-variable-names-from-meta-graph
imported_graph = tf.get_default_graph()
graph_op = imported_graph.get_operations()
# print([x for x in tf.get_default_graph().get_operations() if x.type == "Placeholder"])

# with open('./model/baseline/NN_weights.txt', 'w') as f:
#     for i in graph_op:
#         f.write(str(i))

# graph_b0 = imported_graph.get_operation_by_name('dense_1_vars/bias')
# graph_b1 = imported_graph.get_operation_by_name('dense_1_vars/bias_1')
# print(graph_b0.values())

# https://stackoverflow.com/questions/42769435/get-variable-does-not-work-after-session-restoration
sess = tf.Session()
# sess.run(tf.global_variables_initializer())
saver.restore(sess, 'F:/volume/0217graphsage/0106/model_output/model')
# saver.restore(sess, tf.train.latest_checkpoint('F:/volume/0217graphsage/0106/model_output'))

# print(imported_graph.get_tensor_by_name('Placeholder:0'))
# print(imported_graph.get_operation_by_name('Placeholder'))

# get weight, https://ithelp.ithome.com.tw/articles/10187786
all_vars = tf.trainable_variables()
# print(all_vars[6].name)
# print(sess.run(all_vars[6]).shape)
# call layer with saved weights
node_pred = custom_Dense(all_vars[5], all_vars[6], all_vars[7], all_vars[8], all_vars[9], all_vars[10])


# create paper pair
def gen_paper(nodes, batch_i):
    n_times = nodes.shape[0]
    index_i = np.where(emb_node == batch_i)[0]
    paper_i_emb = paper_emb[index_i]
    # exclude_i = np.delete(paper_emb, index_i, axis=0)  # exclude row i
    # paper_i_pair = np.array([paper_i_emb.tolist(), ] * (n_times-1))  # repeat rows
    # paper_i_pair = np.concatenate((paper_i_pair, exclude_i), axis=1)
    # exclude_i = np.delete(candidate_paper_emb, index_i, axis=0)
    paper_i_pair = np.tile(paper_i_emb, (len(candidate_ids), 1))  # repeat rows
    paper_i_pair = np.concatenate((paper_i_pair, candidate_paper_emb), axis=1)
    batch_y = all_paper[all_paper['head'] == batch_i]['tail'].values  # find tail ids
    # set 1 at tail index
    # i_cited_index = np.where(np.in1d(emb_node, all_paper[all_edge['head'] == batch_i]['tail'].values))
    # batch_y = np.zeros(n_times-1)
    # batch_y[np.in1d(batch_y, i_cited_index)] = 1
    # yield batch_x, batch_y, np.delete(emb_node, index_i)
    yield paper_i_pair, batch_y


def make_prediction(x):
    with tf.Session() as sess:
        saver.restore(sess, 'F:/volume/0217graphsage/0106/model_output/model')
        all_vars = tf.trainable_variables()
        node_pred = custom_Dense(all_vars[5], all_vars[6], all_vars[7], all_vars[8], all_vars[9], all_vars[10])
        i_prediction = sess.run(tf.nn.sigmoid(node_pred(x))).reshape(len(x), -1)
    # sort classes and output at the same time
    # sorter = np.argsort(i_prediction, axis=1)[::-1][:, :K]  # reverse
    # i_prediction = i_prediction[sorter]
    # classes = np.array(batch_classes)[sorter]
    # https://stackoverflow.com/questions/33140674/argsort-for-a-multidimensional-ndarray
    # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
    new_sorter = i_prediction.argsort(axis=1)[::-1][:, :K]  # sort and select first 150
    batch_classes = np.tile(candidate_ids, (len(x), 1))
    classes = np.take_along_axis(batch_classes, new_sorter, axis=1)
    return classes


if task == 0:
    sess.close()
    # to reduce memory usage, use batch prediction
    # batch_paths = np.random.choice(emb_node, size=emb_node.shape[0])
    # batch_paths = np.random.choice(head_paper_ids, size=300)
    batch_paths = head_paper_ids[:30]
    ans = []
    rs = []
    batch_x, batch_y = [], []
    K = 150
    i = 0

    for batch_i in tqdm(batch_paths):
        x, y = next(gen_paper(emb_node, batch_i))
        # avoid empty answers
        if len(y) > 0:
            batch_x.append(x)
            batch_y.append(y.tolist())

        if len(batch_x) > 9:
            batch_x_arr = np.array(batch_x)
            classes = make_prediction(batch_x_arr)  # shape: len(x) * K
            del batch_x_arr
            ans.extend(batch_y)
            rs.extend(classes.tolist())
            batch_x, batch_y = [], []  # reset

        # batch_classes.append(candidate_ids)

    # 先把所有 pair分批產好並存到disk, 之後再做 predict
    np.save('npy_temp/batch300_x_emb.npy', np.array(batch_x))
    np.save('npy_temp/batch300_y_label.npy', np.array(batch_y))

    # calculate MAP
    # print(metrics.mapk(ans, rs, K))


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
        num_neg_sample = num_cited  # 設定negative sample數量
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
    print(np.sum(np.equal(prediction, y))/ len(y))


