import pandas as pd
import numpy as np
import pickle
from keras.utils.vis_utils import plot_model
import graphviz
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import ml_metrics as metrics
import matplotlib.pyplot as plt
import ast
from collections import Counter
from sklearn import preprocessing
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization

os.environ["PATH"] += os.pathsep + 'C:/Users/Wade/Anaconda3/Library/bin/graphviz'

bert_title = 768
bert_abstract = 768
emb_dim = 100  # output dim (就是graph embedding後的dim)

dblp_top50 = pd.read_pickle('preprocess/edge/paper_2011up_venue50.pkl')  # 有舊的paper_id跟references
comparison = pd.Series(dblp_top50.new_papr_id.values, index=dblp_top50.id).to_dict()  # 新舊id對照表
dblp_remain = pd.read_csv('preprocess/edge/dblp_remain_references.csv')  # references到的paper有在我們的pool裡
dblp_remain = dblp_remain.loc[dblp_remain.references_amount > 0]  # 留有ref的
dblp_remain['paper_id'] = dblp_remain.paper_id.map(comparison)  # 將舊id換成新的, map is much faster than using replace
dblp_remain['paper_references'] = dblp_remain.paper_references.apply(lambda x: list(map(comparison.get, list(map(int, ast.literal_eval(x))))))
remain_paper = dblp_top50.loc[(dblp_top50.new_papr_id.isin(dblp_remain.paper_id.values)) & (dblp_top50.time_step < 284)].new_papr_id.values  # 2018AAAI以前
dblp_top50_conf = pd.read_pickle('preprocess/edge/paper_venue.pkl')
pa = pd.read_pickle('preprocess/edge/p_a_before284_delete_author.pkl')
pa_extra = pd.read_pickle('preprocess/edge/p_a_delete_author.pkl')
pp = pd.read_pickle('preprocess/edge/paper_paper.pkl')
pp['new_cited_papr_id'] = pp['new_cited_papr_id'].astype(int).astype(str)
# 從pp整合出 ref list
paper_refs = pp.groupby(['new_papr_id'])['new_cited_papr_id'].agg([','.join]).reset_index()
# 挑出ref>20的index, 只用他們算MAP
paper_refs = paper_refs[paper_refs.new_papr_id.isin(pp[pp.groupby(['new_papr_id'])['year'].transform('count') > 20].new_papr_id.value_counts().index.tolist())]

# FIXME 一篇有多個作者時當多個training sample?
# https://stackoverflow.com/questions/20067636/pandas-dataframe-get-first-row-of-each-group
dblp_top50_conf['new_first_aId'] = pa.groupby('new_papr_id').first()['new_author_id']  # 取每篇的第一作者
dblp_top50_conf['references'] = dblp_top50[dblp_top50['new_papr_id'].isin(dblp_top50_conf['new_papr_id'].values)]['references']
# select 2018以前全部當train
train2017 = dblp_top50_conf.loc[dblp_top50_conf.time_step < 284, ['new_papr_id', 'new_venue_id', 'new_first_aId', 'references']]
dblp_top50_test = dblp_top50_conf.copy()
dblp_top50_test['new_first_aId'] = pa_extra.groupby('new_papr_id').first()['new_author_id']  # 移除沒有作者的paper
test2018 = dblp_top50_test.loc[dblp_top50_test.time_step == 284, ['new_papr_id', 'new_venue_id', 'new_first_aId', 'references']]
# 可以被推薦的pool
paper_pool = train2017.copy()

# 塞入bert
titles = pd.read_pickle('preprocess/edge/titles_bert.pkl')
abstracts = pd.read_pickle('preprocess/edge/abstracts_bert.pkl')
# normalize column/ feature
titles = preprocessing.scale(np.array(titles.tolist()))
abstracts = preprocessing.scale(np.array(abstracts.tolist()))

# deepwalk
node_id = np.load('preprocess/edge/node_id.npy').astype(int)
# emb_2017 = np.load('preprocess/edge/deep_walk_vec.npy')
with open('preprocess/edge/deep_walk_emb.pkl', 'rb') as f:
    emb_2017 = pickle.load(f)
# emb_2017 = preprocessing.scale(emb_2017)
# line embedding
with open('preprocess/edge/line_1and2.pkl', 'rb') as file:
    line_emb = pickle.load(file)
# graphsage
with open('preprocess/edge/paper_embeddings.pkl', 'rb') as f:
    sage_emb = pickle.load(f)

# 檢查是否所有dblp的node都在embedding裡面
# print(np.isin(train2017.new_papr_id.values, node_id).sum())
# print(np.isin(train2017.new_papr_id.values, np.fromiter(line_emb.keys(), dtype=int)).sum())
# print(np.isin(pa.new_author_id.value_counts().index.values, node_id).sum())
# print(np.isin(dblp_top50_conf.new_venue_id.value_counts().index.values, node_id).sum())

# 刪除沒有embedding的node當train
train2017 = train2017.loc[train2017.new_first_aId.isin(node_id)]
train2017 = train2017.loc[train2017.new_venue_id.isin(node_id)]
# train2017.dropna(subset=['references'], inplace=True)
train2017.dropna(subset=['new_first_aId'], inplace=True)
test2018 = test2018.loc[test2018.new_first_aId.isin(node_id)]
test2018 = test2018.loc[test2018.new_venue_id.isin(node_id)]
test2018.dropna(subset=['new_first_aId'], inplace=True)
test2018.dropna(subset=['references'], inplace=True)


# data generator
def embedding_loader(emb_data, file_len, graph="LINE", batch_size=32, shuffle=1, all=1, test=False):
    """
    準備資料給模型
    :param emb_data:
    :param file_len:
    :param stage: 'train' or 'test'
    :param graph: 'LINE' or 'DeepWalk'
    :param batch_size:
    """
    while True:
        batch_x = []
        batch_y = []
        if test:
            emb_p = []
        batch_paths = np.random.choice(a=file_len, size=batch_size)
        if all:
            np.random.shuffle(file_len)  # 確保不會像choice有重複取到的可能
            batch_paths = file_len
        for batch_i in batch_paths:
            # emb_t = titles.iloc[batch_i]  # 找該paper的title, abstract
            # emb_a = abstracts.iloc[batch_i]
            emb_t = titles[batch_i]
            emb_a = abstracts[batch_i]
            # 根據新paper id 找出 aId, vId
            if not test:
                vId, aId = dblp_top50_conf.loc[batch_i, ['new_venue_id', 'new_first_aId']]
            else:
                vId, aId = dblp_top50_test.loc[batch_i, ['new_venue_id', 'new_first_aId']]
            if graph == 'LINE' or graph == 'GraphSAGE' or graph == "DeepWalk":
                if not test:
                    # 找出該paper的所有資訊
                    emb_p = emb_data[str(batch_i)]  # paper emb
                if shuffle:
                    emb1 = emb_data[str(vId)]
                    emb2 = emb_data[str(int(aId))]
                    # todo 只用BERT看看
                    # emb = np.concatenate((emb_t, emb_a), axis=None)
                    emb = np.concatenate((emb1, emb2, emb_t, emb_a), axis=None)
            if shuffle:
                batch_x += [emb]
            batch_y += [emb_p]
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        yield batch_x, batch_paths, batch_y


x_train, _, paper_emb_train = next(embedding_loader(sage_emb, train2017.new_papr_id.values, 'GraphSAGE', train2017.shape[0]))
_, y_all, paper_emb_all = next(embedding_loader(sage_emb, paper_pool.new_papr_id.values, 'GraphSAGE', paper_pool.shape[0], shuffle=0))
# x_train, _, paper_emb_train = next(embedding_loader(emb_2017, train2017.new_papr_id.values, 'DeepWalk', train2017.shape[0]))
# _, y_all, paper_emb_all = next(embedding_loader(emb_2017, paper_pool.new_papr_id.values, 'DeepWalk', paper_pool.shape[0], shuffle=0))


# https://stackoverflow.com/questions/41888085/how-to-implement-word2vec-cbow-in-keras-with-shared-embedding-layer-and-negative
# 用一個network去逼近embedding
# all_in_one = Input(shape=(emb_dim*2+bert_title+bert_abstract,))
all_in_one = Input(shape=(emb_dim*2,))
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

batch = 1024
train_history = model.fit(x_train, paper_emb_train, batch_size=batch, epochs=50, verbose=2)

# save model
model.save('model/hdf5/model_deepwalk_BN.h5')
# model.save('model/hdf5/model_gSAGE_BN.h5')

# 查看weight
print(len(model.layers))
first_layer_weights = model.layers[1].get_weights()[0]  # d1的weight
first_layer_bias = model.layers[1].get_weights()[1]


# plot loss
def show_train_history(train_history, train, validation=''):
    plt.plot(train_history.history[train])
    # plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend()
    # plt.legend(['train', 'validation'], loc='center right')
    plt.show()


show_train_history(train_history, 'loss')

# FIXME testing的embedding還沒有答案
test_history = model.evaluate_generator(embedding_loader(line_emb, test2018.new_papr_id.values, all=0), steps=test2018.shape[0]/ batch)


# 以pp裡面的ref為準看MAP
y = train2017[train2017.new_papr_id.isin(paper_refs['new_papr_id'].values)].new_papr_id.values
# y = test2018.new_papr_id.values  # testing
# x_all, shuffled_y, paper_emb = next(embedding_loader(sage_emb, y, 'GraphSAGE', y.shape[0], test=True))  # testing
x_all, shuffled_y, paper_emb = next(embedding_loader(sage_emb, y, 'GraphSAGE', y.shape[0]))  # 裡面會shuffle過
# x_all, shuffled_y, paper_emb = next(embedding_loader(emb_2017, y, 'DeepWalk', y.shape[0]))
sorter = shuffled_y.argsort()
sort_y = shuffled_y[sorter]
x_all = x_all[sorter]  # 所有input
paper_emb = paper_emb[sorter]  # 100維的paper_emb
K = 150


model = load_model('model/hdf5/model_gSAGE_BN.h5')
# 一次load全部data做predict
predictions = model.predict(x_all, workers=4)  # 預測paper_emb
np.save('model/baseline/tmp/embedding_predictions.npy', predictions)
# predictions = np.load('model/baseline/tmp/embedding_predictions.npy')
# 找出跟NN predict出最相似的embedding當作要推薦cite的論文
neigh = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree', n_jobs=4)  # algorithm='kd_tree', will use less memory
neigh.fit(paper_emb_all, y_all)
papers = neigh.classes_  # 找出index所代表的paper_id

# 看embedding答案的效果
k_predictions_graph = neigh.kneighbors(X=paper_emb, n_neighbors=K, return_distance=False)
k_predictions = neigh.kneighbors(X=predictions, n_neighbors=K, return_distance=False)
recommend_papers = np.zeros((k_predictions.shape[0], k_predictions.shape[1]))
graph_recommend_papers = np.zeros((k_predictions.shape[0], k_predictions.shape[1]))
i = 0
for row in k_predictions_graph:
    graph_recommend_papers[i] = papers[row]  # 從KNN index轉成paper_id
    i += 1
i = 0
for row in k_predictions:
    recommend_papers[i] = papers[row]  # 從KNN index轉成paper_id
    i += 1

# dot for graphSAGE
graph_scores = np.dot(paper_emb_all, paper_emb.T)
graph_recommend_papers = y_all[np.argsort(graph_scores, axis=0)[::-1]][1:K+1]
scores = np.dot(paper_emb_all, predictions.T)
recommend_papers = y_all[np.argsort(scores, axis=0)[::-1]][1:K+1]

# cosine for deepwalk
cos_graph = cosine_similarity(paper_emb_all, paper_emb)
graph_recommend_papers = y_all[np.argsort(cos_graph, axis=0)[::-1]][1:K+1]
cos = np.dot(paper_emb_all, predictions.T)
recommend_papers = y_all[np.argsort(cos, axis=0)[::-1]][:K]

# 算MAP
# ans = train2017.sort_values(by=['new_papr_id']).reset_index(drop=True)
# ansK = ans.references.apply(lambda x: list(filter(None.__ne__, list(map(comparison.get, map(int, ast.literal_eval(x)))))))
# ansK = ans.paper_references.apply(lambda x: x[:K]).values
# print(metrics.mapk(ans1.reshape((-1, 1)).tolist(), first_predictions.reshape((-1, 1)).tolist(), 1))

ans = paper_refs[paper_refs.new_papr_id.isin(train2017['new_papr_id'].values)]  # pp資料的答案
ansK = ans['join'].apply(lambda x: list(map(int, x.split(','))))
# print(metrics.mapk(ansK.tolist(), graph_recommend_papers.T.tolist(), K))

# testing
test_ans = test2018.sort_values(by=['new_papr_id']).reset_index(drop=True)
test_ansK = test_ans.references.apply(lambda x: list(filter(None.__ne__, list(map(comparison.get, map(int, ast.literal_eval(x)))))))
print(metrics.mapk(test_ansK.tolist(), recommend_papers.T.tolist(), K))

# 產生hot
all_refs = []
for i, data in dblp_remain.paper_references.iteritems():
    all_refs.extend(data)
all_refs = Counter(all_refs)
# hot_1 = all_refs.most_common(1)[0][0]
hot = [m[0] for m in all_refs.most_common(K)]

# testing hot
paper284_ids = np.append(train2017.new_papr_id.values, test2018.new_papr_id.values)
hot284 = dblp_remain[dblp_remain['paper_id'].isin(paper284_ids)].paper_references
test_refs = []
for i, data in hot284.iteritems():
    test_refs.extend(data)
test_refs = Counter(test_refs)
test_hot = [m[0] for m in test_refs.most_common(K)]
# test hot
# print(metrics.mapk(test_ansK.tolist(), np.array([test_hot]*test_ansK.shape[0]).tolist(), K))


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


def metric_bar(hot, ansK, method='MAP', t='train', kList=(25, 50, 100, 150)):
    for k in kList:
        if method == 'MAP':
            plt.bar('Hot', metrics.mapk(ansK.tolist(), np.array([hot]*ansK.shape[0]).tolist(), k))
            plt.bar('RS', metrics.mapk(ansK.tolist(), recommend_papers.T.tolist(), k))
            plt.bar('GR', metrics.mapk(ansK.tolist(), graph_recommend_papers.T.tolist(), k))
        if method == 'Recall':
            plt.bar('Hot', mark(ansK.values, np.array([hot]*ansK.shape[0]), k))
            plt.bar('RS', mark(ansK.values, recommend_papers.T, k))
            plt.bar('GR', mark(ansK.values, graph_recommend_papers.T, k))
        plt.ylabel('score')
        plt.title(t+' '+method+'@'+str(k))
        plt.show()


metric_bar(hot, ansK)  # train MAP
metric_bar(test_hot, test_ansK, t='test')  # test MAP
metric_bar(hot, ansK, 'Recall')  # train Recall
