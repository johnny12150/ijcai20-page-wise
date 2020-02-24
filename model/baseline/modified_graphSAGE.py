import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import ml_metrics as metrics
from model.baseline.graphsage_dnn.Layers import Dense

task = 1
with open('F:/volume/0217graphsage/0106/all_edge.pkl', 'rb') as file:
    all_edge = pickle.load(file)

# fixme: df原始 id map到 graphsage node的 id
with open('F:/volume/0217graphsage/0106/id_map.pkl', 'rb') as file:
    id_map = pickle.load(file)

# all_edge['head'] = all_edge['head'].map(id_map)
# all_edge['tail'] = all_edge['tail'].map(id_map)

# filter out relation = 0 (reference)
all_paper = all_edge.loc[all_edge.rel == 0]
all_paprr_id = list(set(all_paper['head'].tolist() + all_paper['tail'].tolist()))
candidate_ids = list(set(all_paper['tail'].tolist()))
# 只有出現在head過的paper
head_paper_ids = list(set(all_paper['head'].tolist()))

# load paper embedding
paper_emb = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/embedding.npy')
emb_node = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/emb_node.npy')
# keep node and emb only in paper_id
index = np.where(np.in1d(emb_node, all_paprr_id))
emb_node = emb_node[np.in1d(emb_node, index)]
paper_emb = paper_emb[index]
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
saver.restore(sess, 'F:/volume/0217graphsage/0106/model_output/model')
# saver.restore(sess, tf.train.latest_checkpoint('F:/volume/0217graphsage/0106/model_output'))

print(sess.run('dense_1_vars/bias'))

# print(imported_graph.get_tensor_by_name('Placeholder:0'))
# print(imported_graph.get_operation_by_name('Placeholder'))
# sess.run(tf.global_variables_initializer())

# get weight, https://ithelp.ithome.com.tw/articles/10187786
all_vars = tf.trainable_variables()
# print(all_vars[6].name)
# print(sess.run(all_vars[6]).shape)
# call layer with saved weights
node_pred = Dense(all_vars[5], all_vars[6], all_vars[7], all_vars[8], all_vars[9], all_vars[10], act=lambda x: x)

# sample = np.zeros((2, 200)).astype('float32')
# node_preds = node_pred(sample)
# print(node_preds.eval(session=sess))

# print(np.where(emb_node == 1))  # where is the paper id
# print(paper_emb[np.where(emb_node == 1)[0]][0].shape)  # get emb by paper id


# create paper pair
def gen_paper(nodes, batch_i):
    n_times = nodes.shape[0]
    index_i = np.where(emb_node == batch_i)[0]
    paper_i_emb = paper_emb[index_i]
    # exclude_i = np.delete(paper_emb, index_i, axis=0)  # exclude row i
    # todo 考慮不排除自己
    # paper_i_pair = np.array([paper_i_emb.tolist(), ] * (n_times-1))  # repeat rows
    # paper_i_pair = np.concatenate((paper_i_pair, exclude_i), axis=1)
    # exclude_i = np.delete(candidate_paper_emb, index_i, axis=0)
    paper_i_pair = np.array([paper_i_emb.tolist(), ] * len(candidate_ids)).reshape(-1, 100)  # repeat rows
    paper_i_pair = np.concatenate((paper_i_pair, candidate_paper_emb), axis=1)
    batch_y = all_paper[all_edge['head'] == batch_i]['tail'].values  # find tail ids
    # set 1 at tail index
    # i_cited_index = np.where(np.in1d(emb_node, all_paper[all_edge['head'] == batch_i]['tail'].values))
    # batch_y = np.zeros(n_times-1)
    # batch_y[np.in1d(batch_y, i_cited_index)] = 1
    batch_x = paper_i_pair
    # yield batch_x, batch_y, np.delete(emb_node, index_i)
    yield batch_x, batch_y, candidate_ids


if task == 0:
    # to reduce memory usage, use batch prediction
    # batch_paths = np.random.choice(emb_node, size=emb_node.shape[0])
    batch_paths = np.random.choice(emb_node, size=100)
    ans = []
    rs = []
    batch_x, batch_y, batch_classes = [], [], []
    acc = 0
    K = 150
    i = 0

    # todo batch_path 應該只考慮有出現在 head的paper
    for batch_i in tqdm(batch_paths):
        x, y, classes = next(gen_paper(emb_node, batch_i))
        if len(y) > 0:
            batch_x.append(x)
            batch_y.append(y.tolist())
            batch_classes.append(classes)
        # avoid empty answers
        if len(batch_x) > 4:
            batch_x_arr = np.array(batch_x).astype('float32').reshape(-1, 200)
            i_prediction = sess.run(node_pred(batch_x_arr)).reshape(len(batch_x), -1)
            del batch_x_arr
            # sort classes and output at the same time
            # sorter = np.argsort(i_prediction, axis=1)[::-1][:, :K]  # reverse
            # i_prediction = i_prediction[sorter]
            # classes = np.array(batch_classes)[sorter]
            # https://stackoverflow.com/questions/33140674/argsort-for-a-multidimensional-ndarray
            # https://stackoverflow.com/questions/20103779/index-2d-numpy-array-by-a-2d-array-of-indices-without-loops
            new_sorter = i_prediction.argsort(axis=1)[::-1][:, :K]  # sort and select first 150
            classes = np.take_along_axis(np.array(batch_classes), new_sorter, axis=1)
            ans.extend(batch_y)
            rs.extend(classes.tolist())
            batch_x, batch_y, batch_classes = [], [], []  # reset

            # calculate accuracy
            # 只算答案是1的部分，看model有沒有train起來
            # ans_len = y.shape[0]
            # i_acc = np.sum(np.isin(classes[:ans_len], y)) / ans_len
            # acc = (acc + i_acc) / len(ans)

    # calculate MAP
    print(metrics.mapk(ans, rs, K))
    print(acc)


# check graphsage dnn
def sage_nn(nodes, save=True):
    x = []
    y = []
    for batch_i in tqdm(nodes):
        # 找node_i真實有site的, 讓nn判斷
        index_i = np.where(emb_node == batch_i)[0]
        paper_i_emb = paper_emb[index_i]  # paper i embedding
        cite_paper = all_paper[all_edge['head'] == batch_i]['tail'].values
        i_cited_index = np.where(np.in1d(emb_node, cite_paper))
        paper_i_cite_emb = paper_emb[i_cited_index]  # paper i 有cite的embedding
        num_cited = paper_i_cite_emb.shape[0]  # paper i cite 幾篇
        # add negative sample
        num_neg_sample = num_cited  # 設定negative sample數量
        neg_index = np.where(~np.in1d(candidate_ids, cite_paper))[0]
        # negative index就是positive的差集
        # neg_index = list(set(range(len(emb_node))) - set(i_cited_index[0].tolist()))
        neg_index = np.random.choice(neg_index, num_neg_sample)  # random select
        neg_sample_emb = paper_emb[neg_index]
        if num_cited > 0:
            repeat_emb = np.tile(paper_i_emb, num_cited).reshape(-1, 100)  # clone rows
            x.extend(np.concatenate((repeat_emb, paper_i_cite_emb), axis=1))  # positive sample
            x.extend(np.concatenate((np.tile(paper_i_emb, num_neg_sample).reshape(-1, 100), neg_sample_emb), axis=1))  # add negative sample
            y.extend(np.ones(num_cited).tolist())
            y.extend(np.zeros(num_neg_sample).tolist())

    if save:
        np.save('./paper_pair.npy', np.array(x))
        np.save('./pair_label.npy', np.array(y))
    return np.array(x), np.array(y)


x, y = sage_nn(head_paper_ids)
y_logit = y.astype('float32').reshape(-1, 1)
prediction_logit = sess.run(node_pred(x.astype('float32')))  # predict
loss = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(logits=prediction_logit, labels=y_logit))
print(np.sum(loss.reshape(-1))/ len(loss))
prediction = sess.run(tf.nn.sigmoid(prediction_logit))
prediction = np.round(prediction.reshape(-1))  # flatten & to 0/ 1
print(np.sum(np.equal(prediction, y))/ len(y))


