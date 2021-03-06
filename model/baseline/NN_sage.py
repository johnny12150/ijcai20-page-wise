import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization, Embedding, concatenate, Reshape
from keras.callbacks import EarlyStopping
import tensorflow as tf
from model.baseline.graphsage_dnn.Layers import custom_Dense, zeros
from model.baseline.eval_metrics import show_average_results
import os
import ml_metrics as metrics

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train = False
GNN = 'GraphSAGE'
bert_title = 768
bert_abstract = 768
emb_dim = 100  # output dim (就是graph embedding後的dim)

pv = pd.read_pickle('preprocess/edge/paper_venue.pkl')
pa_extra = pd.read_pickle('preprocess/edge/p_a_delete_author.pkl')
# 塞入bert
titles = pd.read_pickle('preprocess/edge/titles_bert.pkl')
abstracts = pd.read_pickle('preprocess/edge/abstracts_bert.pkl')
# normalize column/ feature
titles = preprocessing.scale(np.array(titles.tolist()))
abstracts = preprocessing.scale(np.array(abstracts.tolist()))


def gen_paper(paper_i_emb, t):
    paper_i_pair = np.tile(paper_i_emb, (len(candidate_paper_emb))).reshape(-1, 100)  # repeat rows
    candidates = np.tile(candidate_paper_emb, (len(paper_i_emb), 1))  # repeat candidate_ids
    paper_i_pair = np.concatenate((paper_i_pair, candidates), axis=1)  # shape: N * len(candidate_ids) * 200
    return graphsage_nn(paper_i_pair, len(paper_i_emb), t=t)


# 用 graphsage train的 nn來推薦
def graphsage_nn(x, size=20, K=150, t=281):
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
            saver.restore(sess, 'F:/volume/0217graphsage/0106/rolling_models/'+str(t)+'/model')
            # saver.restore(sess, 'F:/volume/0217graphsage/0106/model_output/model')
            node_pred = custom_Dense(w0, w1, w2, b0, b1, b2)
            pred = sess.run(tf.nn.sigmoid(node_pred(x.astype('float32')))).reshape(size, -1)

    # sort and select first 150
    new_sorter = pred.argsort(axis=1)[:, -K:]  # 取值最大的150個
    new_sorter = np.flip(new_sorter, axis=1)  # 前面取到的最大值
    batch_classes = np.tile(candidate_ids, (size, 1))  # shape: N * len(candidate_ids)
    classes = np.take_along_axis(batch_classes, new_sorter, axis=1)
    return classes


def graph_bert_pair(save=True):
    x, y, x_ids, y_label = [], [], [], []
    for i in tqdm(head_paper_ids):
        p_emb = paper_emb[np.where(emb_node == i)[0]]
        refs = all_paper[all_paper['head'] == i]['tail'].values
        # 找 i相關的 BERT emb
        emb_t = titles[i]
        emb_a = abstracts[i]
        # authors = all_edge[(all_edge['head']==i) & (all_edge.rel==1)]['tail'].values  # author
        authors = pa_extra[pa_extra['new_papr_id'] == i]['new_author_id'].values
        # authors = list(map(id_map.get, authors))
        v = pv[pv['new_papr_id'] == i]['new_venue_id'].values
        # v = list(map(id_map.get, v))
        if len(v) == 0:
            # v_emb = np.zeros(emb_dim)
            continue
        else:
            v_emb = all_emb[np.where(np.in1d(all_node, v))]
            if len(v_emb) == 0:
                # v_emb = np.zeros(emb_dim)
                continue
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
            repeat = np.tile(concat, (len(a_emb), 1))
            emb_concat = np.concatenate((a_emb, repeat), axis=1)
            x.extend(emb_concat)
            repeat_i = np.tile(p_emb, (len(a_emb), 1))
            y.extend(repeat_i)
            x_ids.extend(np.tile(i, (len(a_emb), 1)).tolist())
            y_label.extend(np.tile(refs, (len(a_emb), 1)).tolist())
        else:
            emb_concat = np.concatenate((a_emb, v_emb, emb_t, emb_a), axis=None)
            x += [emb_concat]
            y += [p_emb.reshape(-1)]
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
    # 分開 input後過 embedding
    abs = Input(shape=(bert_abstract,))
    til = Input(shape=(bert_title,))
    au = Input(shape=(emb_dim,))
    ve = Input(shape=(emb_dim,))
    emb1 = Dense(bert_abstract)(abs)
    emb2 = Dense(bert_title)(til)
    emb3 = Dense(emb_dim)(au)
    emb4 = Dense(emb_dim)(ve)
    con = concatenate([emb1, emb2, emb3, emb4])
    d1 = Dense(2000, activation='tanh')(con)
    d2 = Dense(1000, activation='tanh')(d1)
    d3 = Dense(5000, activation='tanh')(d2)
    out_emb = Dense(emb_dim, activation='linear')(d3)
    model = Model([au, ve, til, abs], out_emb)

    # d1 = Dense(2000, activation='tanh')(BN)
    # # d1 = BatchNormalization()(d1)
    # d2 = Dense(1000, activation='tanh')(d1)
    # # d2 = BatchNormalization()(d2)
    # d3 = Dense(800, activation='tanh')(d2)
    # d4 = Dense(500, activation='tanh')(d3)
    # d5 = Dense(200, activation='tanh')(d4)
    # out_emb = Dense(emb_dim, activation='linear')(d5)
    # model = Model(input=all_in_one, output=out_emb)

    print(model.summary())

    model.compile(optimizer='adam', loss='mae')
    # model.compile(optimizer='adam', loss='cosine_proximity')
    # plot_model(model, to_file='pics/model_LINE.png', show_shapes=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    # train_history = model.fit(x, y, batch_size=batch, epochs=30, verbose=2, validation_split=0.33, callbacks=[early_stopping])
    train_history = model.fit([x[:, :100], x[:, 100:200], x[:, 200:968], x[:, 968:]], y, batch_size=batch, epochs=30, verbose=2, validation_split=0.33, callbacks=[early_stopping])

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


def pred_paper_emb(x_all):
    model = load_model('model/hdf5/model_gSAGE_NN.h5')
    predictions = model.predict([x_all[:, :100], x_all[:, 100:200], x_all[:, 200:968], x_all[:, 968:]], workers=4)
    return predictions


def gen_test_data(test_nodes, t, rec=True, rolling=False):
    x, y = [], []
    for i in test_nodes:
        ans = pp[pp['new_papr_id'] == i]['new_cited_papr_id'].values
        if not rolling:
            if len(ans) > 0:
                emb_title = titles[i]
                emb_abs = abstracts[i]
                author = pa_extra[pa_extra['new_papr_id'] == i]['new_author_id'].values
                venue = pv[pv['new_papr_id'] == i]['new_venue_id'].values
                emb_venue = all_emb[np.where(np.in1d(all_node, venue))]
                if len(author) > 0:
                    if len(emb_venue) == 0:
                        # emb_venue = np.zeros(emb_dim)
                        continue
                    emb_author = all_emb[np.where(np.in1d(all_node, author))]
                    concat = np.concatenate((emb_venue, emb_title, emb_abs), axis=None)
                    if len(emb_author) > 1:
                        repeat = np.tile(concat, (len(emb_author), 1))
                        emb_concat = np.concatenate((emb_author, repeat), axis=1)
                        x.extend(emb_concat.tolist())
                        repeat_y = np.tile(ans, (len(emb_author), 1))
                        y.extend(repeat_y.tolist())
                    elif len(emb_author) == 1:
                        x.append(np.concatenate((emb_author.reshape(-1), concat), axis=0).tolist())
                        y.append(ans.tolist())
        else:
            if len(ans) > 0:
                try:
                    x.append(sage_emb_dnn[i])
                    y.append(ans.tolist())
                except:
                    continue
    print('測試篇數為'+str(len(y)))
    if not rec:
        return y
    else:
        if not rolling:
            # 先預測出 paper emb
            p_emb = pred_paper_emb(np.array(x))
        else:
            # 直接load graphsage出的embedding (只考慮pa, pv的)
            p_emb = np.array(x)

        results = []
        batch_size = 20
        for j in tqdm(range(0, len(x), batch_size)):
            if j+batch_size > len(x):
                recommend = gen_paper(p_emb[j:], t).tolist()
                ans = y[j:]
                results.extend(recommend)
            else:
                recommend = gen_paper(p_emb[j:(j+batch_size)], t).tolist()
                ans = y[j:(j+batch_size)]
                results.extend(recommend)
        return results, y


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
    test_timesteps = [281, 284, 302, 307, 310, 318, 321]
    for ts in test_timesteps:
        # 根據時間 load embedding
        all_emb = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/embedding'+str(ts)+'.npy')
        all_node = np.load('F:/volume/0217graphsage/0106/author_venue_embedding/emb_node'+str(ts)+'.npy')
        sort_ = np.argsort(all_node)
        all_node = all_node[sort_]
        all_emb = all_emb[sort_]
        # convert two 1d array to dict
        sage_emb_dnn = {}
        for id, emb in zip(all_node, all_emb):
            sage_emb_dnn[id] = emb

        # todo 考慮改成直接從 pp裡面找
        with open('F:/volume/0217graphsage/0106/all_edge_'+str(ts)+'.pkl', 'rb') as file:
            all_edge = pickle.load(file)
        all_paper = all_edge.loc[all_edge.rel == 0]
        all_paper_id = np.sort(np.array(list(set(all_paper['head'].tolist() + all_paper['tail'].tolist()))))
        candidate_ids = np.sort(np.array(list(set(all_paper['tail'].tolist()))))
        # 只有出現在head過的paper
        head_paper_ids = sorted(list(set(all_paper['head'].tolist())))
        index = np.where(np.in1d(all_node, all_paper_id))
        emb_node = all_node[index]
        paper_emb = all_emb[index]
        # find candidate papers' embedding
        c_index = np.where(np.in1d(emb_node, candidate_ids))
        candidate_paper_emb = paper_emb[c_index]

        timestep = pv.loc[pv.time_step == ts, 'new_papr_id']
        test_rec, label = gen_test_data(timestep.values, ts, rolling=True)
        if len(label) > 0 and len(test_rec) > 0:
            print('RS, t= ' + str(ts))
            show_average_results(label, test_rec)
        print('-'*30)
