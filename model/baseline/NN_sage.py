import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import ast
from collections import Counter
from sklearn import preprocessing
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization
import tensorflow as tf
from model.baseline.graphsage_dnn.Layers import custom_Dense, zeros
from model.baseline.eval_metrics import show_average_results


train = False
GNN = 'GraphSAGE'
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
# 把df原始 id map到 graphsage node的 id
# with open('F:/volume/0217graphsage/0106/id_map.pkl', 'rb') as file:
#     id_map = pickle.load(file)
# all_edge['head'] = all_edge['head'].map(id_map)
# all_edge['tail'] = all_edge['tail'].map(id_map)

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


def gen_paper(paper_i_emb, label):
    # fixme 要用batch的方式避免一次太多篇
    paper_i_pair = np.tile(paper_i_emb, (len(candidate_ids))).reshape(-1, 100)  # repeat rows
    candidates = np.tile(candidate_paper_emb, (len(paper_emb))).reshape(-1, 100)  # repeat candidate_ids
    paper_i_pair = np.concatenate((paper_i_pair, candidates), axis=1)  # shape: N * len(candidate_ids) * 200
    # repeat ans with len(candidate_ids) times
    # label = np.tile(label, (len(candidate_ids), 1))

    # todo predict
    graphsage_nn()
    return paper_i_pair


# 用 graphsage train的 nn來推薦
def graphsage_nn(x, size=20, K=150):
    with tf.Graph().as_default():
        w0 = tf.get_variable('dense_1_vars/weights', shape=(200, 50))  # define variables that we want
        w1 = tf.get_variable('dense_1_vars/weights_1', shape=(50, 50))
        w2 = tf.get_variable('dense_1_vars/weights_2', shape=(50, 1))
        b0 = zeros(50, 'dense_1_vars/bias')
        b1 = zeros(50, 'dense_1_vars/bias_1')
        b2 = zeros(1, 'dense_1_vars/bias_2')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, 'F:/volume/0217graphsage/0106/model_output/model')
            node_pred = custom_Dense(w0, w1, w2, b0, b1, b2)
            pred = sess.run(tf.nn.sigmoid(node_pred(x.astype('float32'))))
    new_sorter = pred.argsort(axis=1)[::-1][:, :K]  # sort and select first 150
    batch_classes = np.tile(candidate_ids, (size, 1))  # shape: N * len(candidate_ids)
    classes = np.take_along_axis(batch_classes, new_sorter, axis=1)
    return classes


def graph_bert_pair(save=True):
    x, y, x_ids, y_label = [], [], [], []
    for i in tqdm(head_paper_ids):
        # head = paper_emb[np.where(np.in1d(emb_node, i))[0]]
        refs = all_paper[all_paper['head'] == i]['tail'].values
        # 找 i相關的 BERT emb
        emb_t = titles[i]
        emb_a = abstracts[i]
        authors = all_edge[(all_edge['head']==i) & (all_edge.rel==1)]['tail'].values  # author
        v = all_edge[(all_edge['head']==i) & (all_edge.rel==8)]['tail'].values  # venue
        if len(v) == 0:
            v_emb = np.zeros(emb_dim)
        else:
            v_emb = all_emb[np.where(np.in1d(all_node, v))]
        if len(authors) == 0:
            a_emb = np.zeros(emb_dim)
        else:
            a_emb = all_emb[np.where(np.in1d(all_node, authors))]
            # 如果都找不到作者embedding
            if len(a_emb) == 0:
                a_emb = np.zeros((len(authors), emb_dim))
        # 若多個作者則拆成多筆
        if len(authors) > 1:
            concat = np.concatenate((v_emb, emb_t, emb_a), axis=None)
            repeat = np.tile(concat, len(a_emb))
            emb_concat = np.concatenate((a_emb, repeat), axis=None)
            x.extend(emb_concat.reshape(len(a_emb), -1))
            repeat_i = np.tile(paper_emb[np.where(emb_node == i)[0]], len(a_emb)).reshape(len(a_emb), -1)
            y.extend(repeat_i)
            x_ids.extend(np.tile(i, (len(a_emb), 1)).tolist())
            y_label.extend(np.tile(refs, (len(a_emb), 1)).tolist())
        else:
            emb_concat = np.concatenate((a_emb, v_emb, emb_t, emb_a), axis=None)
            x += [emb_concat]
            y += [paper_emb[np.where(emb_node == i)[0]].reshape(-1)]
            x_ids.append([i])
            y_label.append(refs.tolist())
    if save:
        np.save('./nn_embs.npy', np.array(x))
        np.save('./nn_label.npy', np.array(y))
    return np.array(x), np.array(y), x_ids, y_label


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

    # model.compile(optimizer='adam', loss='mae')
    model.compile(optimizer='adam', loss='cosine_proximity')
    # plot_model(model, to_file='pics/model_LINE.png', show_shapes=True)

    train_history = model.fit(x, y, batch_size=batch, epochs=30, verbose=2, validation_split=0.33)

    if save:
        model.save('model/hdf5/model_gSAGE_NN.h5')
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


def gen_test_data(test_nodes):
    x, y = [], []
    for i in tqdm(test_nodes):
        ans = pp[pp['new_papr_id'] == i]['new_cited_papr_id'].values
        emb_title = titles[i]
        emb_abs = abstracts[i]
        author = pa_extra[pa_extra['new_papr_id'] == i]['new_author_id'].values
        venue = pv[pv['new_papr_id'] == i]['new_venue_id'].values
        emb_venue = all_emb[np.where(np.in1d(all_node, venue))]
        if len(author) > 0:
            # fixme handle missing venue embedding
            emb_author = all_emb[np.where(np.in1d(all_node, author))]
            concat = np.concatenate((emb_venue, emb_title, emb_abs), axis=None)
            if len(emb_author) > 1:
                repeat = np.tile(concat, len(emb_author))
                emb_concat = np.concatenate((emb_author, repeat), axis=None)
                x.extend(emb_concat.reshape(len(emb_author), -1).tolist())
                repeat_y = np.tile(ans, (len(emb_author), 1))
                y.extend(repeat_y.tolist())
            elif len(emb_author) == 1:
                x.append(np.concatenate((emb_author, concat), axis=None).tolist())
                y.append(ans.tolist())

    results = []
    batch_size = 20
    result = gen_paper()
    return results


if train:
    x, y, x_ids, y_label = graph_bert_pair()
    # shuffle x, y
    # p = np.random.permutation(len(x))
    # x, y = x[p], y[p]
    model, train_history = train_model(x, y, save=True)
    show_train_history(train_history, 'loss', 'val_loss')
else:
    # 找特定時間的 paper id
    pp = pd.read_pickle('preprocess/edge/paper_paper.pkl')
    pv = pd.read_pickle('preprocess/edge/paper_venue.pkl')
    pa_extra = pd.read_pickle('preprocess/edge/p_a_delete_author.pkl')
    # dblp_top50 = pd.read_pickle('preprocess/edge/paper_2011up_venue50.pkl')
    # pv['authors'] = pa_extra.groupby('new_papr_id')['new_author_id'].apply(list)  # groupby element to list
    # pv['references'] = dblp_top50[dblp_top50['new_papr_id'].isin(pv['new_papr_id'].values)]['references']
    # pv.dropna(subset=['authors'], inplace=True)  # drop empty author papers
    # pv = pd.DataFrame([np.append(row.values, d) for _, row in pv.iterrows() for d in row['authors']], columns=pv.columns.append(pd.Index(['new_first_aId'])))
    test_timesteps = [284, 302, 307, 310, 318, 321]
    for ts in test_timesteps:
        # test_timestep = pv.loc[pv.time_step == ts, ['new_papr_id', 'new_venue_id', 'new_first_aId', 'references']]
        timestep = pv.loc[pv.time_step == ts, 'new_papr_id']
        test_emb, label = gen_test_data(timestep.values)


