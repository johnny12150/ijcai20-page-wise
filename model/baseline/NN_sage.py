import pandas as pd
import numpy as np
import pickle
from keras.utils.vis_utils import plot_model
import graphviz
from tqdm import tqdm
import os
import ml_metrics as metrics
import matplotlib.pyplot as plt
import ast
from collections import Counter
from sklearn import preprocessing
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization
import tensorflow as tf
from model.baseline.graphsage_dnn.Layers import Dense


os.environ["PATH"] += os.pathsep + 'C:/Users/Wade/Anaconda3/Library/bin/graphviz'

bert_title = 768
bert_abstract = 768
emb_dim = 100  # output dim (就是graph embedding後的dim)

# 塞入bert
titles = pd.read_pickle('preprocess/edge/titles_bert.pkl')
abstracts = pd.read_pickle('preprocess/edge/abstracts_bert.pkl')
# normalize column/ feature
titles = preprocessing.scale(np.array(titles.tolist()))
abstracts = preprocessing.scale(np.array(abstracts.tolist()))

# graphsage_dnn
all_emb = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/embedding.npy')
all_node = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/emb_node.npy')
# convert two 1d to dict
sage_emb_dnn = {}
for id, emb in zip(all_node, all_emb):
    sage_emb_dnn[id] = emb

with open('F:/volume/0217graphsage/0106/all_edge.pkl', 'rb') as file:
    all_edge = pickle.load(file)
all_paper = all_edge.loc[all_edge.rel == 0]
all_paper_id = list(set(all_paper['head'].tolist() + all_paper['tail'].tolist()))
candidate_ids = list(set(all_paper['tail'].tolist()))
# 只有出現在head過的paper
head_paper_ids = list(set(all_paper['head'].tolist()))

index = np.where(np.in1d(all_node, all_paper_id))
emb_node = all_node[index]
paper_emb = all_emb[index]
# find candidate papers' embedding
c_index = np.where(np.in1d(emb_node, candidate_ids))
candidate_paper_emb = paper_emb[c_index]


def gen_paper(nodes, batch_i):
    n_times = nodes.shape[0]
    index_i = np.where(emb_node == batch_i)[0]
    paper_i_emb = paper_emb[index_i]
    paper_i_pair = np.array([paper_i_emb.tolist(), ] * len(candidate_ids)).reshape(-1, 100)  # repeat rows
    paper_i_pair = np.concatenate((paper_i_pair, candidate_paper_emb), axis=1)
    batch_y = all_paper[all_paper['head'] == batch_i]['tail'].values  # find tail ids
    batch_x = paper_i_pair
    yield batch_x, batch_y, candidate_ids


# todo 用 graphsage train的 nn來推薦
# 存predict的emb給 (modified_graphSAGE.py)
def load_graphsage_nn(x):
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph('F:/volume/0217graphsage/0106/model_output/model.meta')
    imported_graph = tf.get_default_graph()
    graph_op = imported_graph.get_operations()
    with tf.Session() as sess:
        saver.restore(sess, 'F:/volume/0217graphsage/0106/model_output/model')
        all_vars = tf.trainable_variables()
        node_pred = Dense(all_vars[5], all_vars[6], all_vars[7], all_vars[8], all_vars[9], all_vars[10],
                          act=lambda x: x)
        pred = sess.run(node_pred(x.astype('float32')))
    return pred


def sage_nn_train_map():
    batch_paths = np.random.choice(head_paper_ids, size=100)
    ans = []
    rs = []
    batch_x, batch_y, batch_classes = [], [], []
    acc = 0
    K = 150
    i = 0

    for batch_i in tqdm(batch_paths):
        x, y, classes = next(gen_paper(emb_node, batch_i))
        if len(y) > 0:
            batch_x.append(x)
            batch_y.append(y.tolist())
            batch_classes.append(classes)
        # avoid empty answers
        if len(batch_x) > 4:
            batch_x_arr = np.array(batch_x).astype('float32').reshape(-1, 200)
            i_prediction = load_graphsage_nn(batch_x_arr)
            # i_prediction = sess.run(node_pred(batch_x_arr)).reshape(len(batch_x), -1)
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

    # calculate MAP
    print(metrics.mapk(ans, rs, K))


def gen_paper2(nodes, batch_i):
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
    batch_y = all_paper[all_paper['head'] == batch_i]['tail'].values  # find tail ids
    # set 1 at tail index
    # i_cited_index = np.where(np.in1d(emb_node, all_paper[all_edge['head'] == batch_i]['tail'].values))
    # batch_y = np.zeros(n_times-1)
    # batch_y[np.in1d(batch_y, i_cited_index)] = 1
    batch_x = paper_i_pair
    # yield batch_x, batch_y, np.delete(emb_node, index_i)
    yield batch_x, batch_y, candidate_ids


def graph_bert_pair(save=True):
    x, y = [], []
    for i in tqdm(head_paper_ids):
        head = paper_emb[np.where(np.in1d(emb_node, i))[0]]
        # 找 i相關的 BERT emb
        emb_t = titles[i]
        emb_a = abstracts[i]
        authors = all_edge[(all_edge['head']==i) & (all_edge.rel==1)]['tail'].values  # author
        v = all_edge[(all_edge['head']==i) & (all_edge.rel==8)]['tail'].values  # venue
        if len(authors) == 0:
            v_emb = np.zeros(emb_dim)
        else:
            v_emb = all_emb[np.where(np.in1d(all_node, v))]
        if len(authors) == 0:
            a_emb = np.zeros(emb_dim)
        else:
            a_emb = all_emb[np.where(np.in1d(all_node, authors))]
        # 若多個作者則拆成多筆
        if a_emb.shape[0] > 1:
            concat = np.concatenate((v_emb, emb_t, emb_a), axis=None)
            repeat = np.tile(concat, len(authors))
            emb_concat = np.concatenate((a_emb, repeat), axis=None)
        else:
            emb_concat = np.concatenate((a_emb, v_emb, emb_t, emb_a), axis=None)
        x += [emb_concat]
        y += [paper_emb[np.where(emb_node == i)[0]]]
        if save:
            np.save('./nn_embs.npy', np.array(x))
            np.save('./nn_label.npy', np.array(y))
        return np.array(x), np.array(y)


x, y = graph_bert_pair()
# shuffle x, y
# p = np.random.permutation(len(x))
# x, y = x[p], y[p]


def train_model(x, y, batch=1024, save=False):
    # https://stackoverflow.com/questions/41888085/how-to-implement-word2vec-cbow-in-keras-with-shared-embedding-layer-and-negative
    # 用一個network去逼近embedding
    all_in_one = Input(shape=(emb_dim*2+bert_title+bert_abstract,))
    BN = BatchNormalization()(all_in_one)
    d1 = Dense(1000, activation='tanh')(BN)
    # d1 = BatchNormalization()(d1)
    d2 = Dense(512, activation='tanh')(d1)
    # d2 = BatchNormalization()(d2)
    d3 = Dense(256, activation='tanh')(d2)
    d4 = Dense(200, activation='tanh')(d3)
    d5 = Dense(128, activation='tanh')(d4)
    out_emb = Dense(emb_dim, activation='linear')(d5)
    model = Model(input=all_in_one, output=out_emb)
    print(model.summary())

    model.compile(optimizer='adam', loss='cosine_proximity')
    # plot_model(model, to_file='pics/model_LINE.png', show_shapes=True)

    train_history = model.fit(x, y, batch_size=batch, epochs=10, verbose=2, validation_split=0.33)

    if save:
        # save model
        # model.save('model/hdf5/model_deepwalk_BN.h5')
        model.save('model/hdf5/model_gSAGE_BN.h5')
    return model, train_history


# plot loss
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend()
    plt.legend(['train', 'validation'], loc='center right')
    plt.show()


model, train_history = train_model(x_train, paper_emb_train)
show_train_history(train_history, 'loss', 'val_loss')
batch = 1024


def generate_ans(data, t='train'):
    if t == 'train':
        # ans = paper_refs[paper_refs.new_papr_id.isin(data['new_papr_id'].values), 'join'].apply(lambda x: list(map(int, x.split(','))))
        ans_t = paper_refs[paper_refs.new_papr_id.isin(data), 'join']
        ans = ans_t.apply(lambda x: list(map(int, x.split(','))))
    else:
        dblp = dblp_top50_test.drop_duplicates(subset=['new_papr_id']).sort_values(by=['new_papr_id']).reset_index(drop=True)
        test_ans = pd.concat([dblp[dblp['new_papr_id'].eq(x)] for x in data], ignore_index=True).fillna(value='[]')
        ans = test_ans.references.apply(lambda x: list(filter(None.__ne__, list(map(comparison.get, map(int, ast.literal_eval(x)))))))
    return ans


ansK = generate_ans(y_all)
# testing 2018 AAAI
test_ansK = generate_ans(sort_y, t='test')


def _ark(actual, predicted, k=10):
    """
    Computes the average recall at k.
    Parameters
    ----------
    actual : list
        A list of actual items to be predicted
    predicted : list
        An ordered list of predicted items
    k : int, default = 10
        Number of predictions to consider
    Returns:
    -------
    score : int
        The average recall at k.
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / len(actual)


def mark(actual, predicted, k=10):
    """
    Computes the mean average recall at k.
    Parameters
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        mark: int
            The mean average recall at k (mar@k)
    """
    return np.mean([_ark(a,p,k) for a,p in zip(actual, predicted)])


def metric_bar(hot, ansK, recommend, method='MAP', t='train', kList=(25, 50, 100, 150)):
    for k in kList:
        if method == 'MAP':
            plt.bar('Hot', metrics.mapk(ansK.tolist(), np.array([hot]*ansK.shape[0]).tolist(), k))
            plt.bar('RS', metrics.mapk(ansK.tolist(), recommend.T.tolist(), k))
            if t == 'train':
                plt.bar('GR', metrics.mapk(ansK.tolist(), graph_recommend_papers.T.tolist(), k))
        if method == 'Recall':
            plt.bar('Hot', mark(ansK, np.array([hot]*ansK.shape[0]), k))
            plt.bar('RS', mark(ansK, recommend.T, k))
            if t == 'train':
                plt.bar('GR', mark(ansK.values, graph_recommend_papers.T, k))
        plt.ylabel('score')
        plt.title(t+' '+method+'@'+str(k))
        plt.show()


# 找答案非空list的做testing, 避免MAP被灌水
not_null = np.nonzero(test_ansK.values)
test_ansK = test_ansK.values[not_null]
recommend_papers = recommend_papers.T[not_null].T

metric_bar(hot, ansK, recommend_papers)  # train MAP
metric_bar(test_hot, test_ansK, recommend_papers, t='test', kList=[150])  # test MAP
metric_bar(hot, ansK, recommend_papers, 'Recall')  # train Recall
metric_bar(test_hot, test_ansK, recommend_papers, 'Recall', t='test', kList=[25, 50, 150])  # test recall

