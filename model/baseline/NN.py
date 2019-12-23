import pandas as pd
import numpy as np
import pickle
from keras.models import Model, load_model
from keras.layers import Input, Dense, BatchNormalization
from keras.utils.vis_utils import plot_model
import graphviz
import os
from sklearn.neighbors import KNeighborsClassifier
import ml_metrics as metrics
import matplotlib.pyplot as plt
import ast
from collections import Counter

os.environ["PATH"] += os.pathsep + 'C:/Users/Wade/Anaconda3/Library/bin/graphviz'

bert_title = 768
bert_abstract = 768
# output dim (就是graph embedding後的dim)
emb_dim = 100

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

# FIXME train的時候只考慮第一作者 ?
# https://stackoverflow.com/questions/20067636/pandas-dataframe-get-first-row-of-each-group
dblp_top50_conf['new_first_aId'] = pa.groupby('new_papr_id').first()['new_author_id']  # 取每篇的第一作者
dblp_top50_conf.dropna(subset=['new_first_aId'], inplace=True)  # 移除沒有作者的paper
# select 2018以前全部當train
train2017 = dblp_top50_conf.loc[dblp_top50_conf.year < 284, ['new_papr_id', 'new_venue_id', 'new_first_aId']]
test2018 = dblp_top50_conf.loc[dblp_top50_conf.year >= 284, ['new_papr_id', 'new_venue_id', 'new_first_aId']]

# 塞入bert FIXME normalize
titles = pd.read_pickle('preprocess/edge/titles_bert.pkl')
abstracts = pd.read_pickle('preprocess/edge/abstracts_bert.pkl')

# https://stackoverflow.com/questions/41888085/how-to-implement-word2vec-cbow-in-keras-with-shared-embedding-layer-and-negative
# 用一個network去逼近embedding
all_in_one = Input(shape=(emb_dim*2+bert_title+bert_abstract,))
BN = BatchNormalization()(all_in_one)
d1 = Dense(512, activation='tanh')(BN)
d2 = Dense(400, activation='tanh')(d1)
d3 = Dense(256, activation='tanh')(d2)
out_emb = Dense(emb_dim, activation='linear')(d3)
model = Model(input=all_in_one, output=out_emb)

# 最後一層用sigmoid/ linear輸出100個units, loss用mae硬做
model.compile(optimizer='rmsprop', loss='mae')
# plot_model(model, to_file='pics/model_LINE.png', show_shapes=True)

# deepwalk, FIXME 對column normalize
node_id = np.load('preprocess/edge/node_id.npy').astype(int)
emb_2017 = np.load('preprocess/edge/deep_walk_vec.npy')
# line embedding
with open('preprocess/edge/line_1and2.pkl', 'rb') as file:
    line_emb = pickle.load(file)

# 檢查是否所有dblp的node都在embedding裡面
# print(np.isin(train2017.new_papr_id.values, node_id).sum())
# print(np.isin(train2017.new_papr_id.values, np.fromiter(line_emb.keys(), dtype=int)).sum())
# print(np.isin(pa.new_author_id.value_counts().index.values, node_id).sum())
# print(np.isin(dblp_top50_conf.new_venue_id.value_counts().index.values, node_id).sum())

# 刪除沒有embedding的node當train
train2017 = train2017.loc[train2017.new_first_aId.isin(node_id)]
train2017 = train2017.loc[train2017.new_venue_id.isin(node_id)]
test2018 = test2018.loc[test2018.new_first_aId.isin(node_id)]
test2018 = test2018.loc[test2018.new_venue_id.isin(node_id)]


# data generator
def embedding_loader(emb_data, file_len, graph="LINE", batch_size=32, shuffle=1):
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
        batch_paths = np.random.choice(a=file_len, size=batch_size)
        for batch_i in batch_paths:
            emb_t = titles.iloc[batch_i]  # 找該paper的title, abstract
            emb_a = abstracts.iloc[batch_i]
            # 根據新paper id 找出 aId, vId
            vId, aId = dblp_top50_conf.loc[batch_i, ['new_venue_id', 'new_first_aId']]
            if graph == 'LINE':
                # 找出該paper的所有資訊
                emb_p = emb_data[str(batch_i)]  # paper emb
                emb1 = emb_data[str(vId)]
                emb2 = emb_data[str(int(aId))]
                emb = np.concatenate((emb1, emb2, emb_t, emb_a), axis=None)
            elif graph == "DeepWalk":
                emb_p = emb_data[np.where(node_id == batch_i)[0][0]]  # find node id 在emb_data的index
                emb_v = np.where(node_id == int(vId))[0][0]
                emb_A = np.where(node_id == int(aId))[0][0]
                emb = np.hstack(emb_data[[emb_v, emb_A], :])  # np style
                emb = np.concatenate((emb, emb_t, emb_a), axis=None)
            batch_x += [emb]
            batch_y += [emb_p]
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        if shuffle:
            yield batch_x, batch_y
        else:
            yield batch_x, batch_paths



batch = 128
# 先用2018前全部 train推薦系統, 2019的test推薦效果
train_history = model.fit_generator(embedding_loader(emb_2017, train2017.new_papr_id.values, 'DeepWalk', batch), epochs=10, steps_per_epoch=train2017.shape[0] / batch, verbose=2)

# train_history = model.fit_generator(embedding_loader(line_emb, train2017.new_papr_id.values), epochs=10, steps_per_epoch=train2017.shape[0] / batch, verbose=2)

# save model
model.save('model/hdf5/model_deepwalk.h5')
model.save('model/hdf5/model_deepwalk_BN.h5')

# 查看weight
print(len(model.layers))
first_layer_weights = model.layers[1].get_weights()[0]  # d1的weight
first_layer_bias = model.layers[1].get_weights()[1]

# FIXME testing的embedding還沒有答案
test_history = model.evaluate_generator(embedding_loader(line_emb, test2018.new_papr_id.values), steps=test2018.shape[0]/ batch)

# 產生 X為embedding, y為id的pair
y = train2017[train2017.new_papr_id.isin(remain_paper)].new_papr_id.values
X = emb_2017[y]  # FIXME 從node id找paper id
x_all, shuffled_y = next(embedding_loader(emb_2017, y, 'DeepWalk', y.shape[0], 0))  # 裡面會shuffle過
sorter = shuffled_y.argsort()
x_all = x_all[sorter]

model = load_model('model/hdf5/model_deepwalk.h5')
# predictions = model.predict_generator(embedding_loader(emb_2017, y, 'DeepWalk', 256), steps=len(y)/ 256)
# 一次load全部data做predict
predictions = model.predict(x_all, workers=4)

# 找出跟NN predict出最相似的embedding當作要推薦cite的論文
neigh = KNeighborsClassifier(n_neighbors=3, n_jobs=4)
neigh.fit(X, y)
first_predictions = neigh.predict(predictions)  # 每row的預測ref的paper id
# 透過機率找出最大的幾篇來推
n_predictions = neigh.predict_proba(predictions)
# print(neigh.classes_)

# 算MAP
# FIXME 答案應該是ref而不是paper id
ans1 = dblp_remain[dblp_remain.isin(y)].sort_values(by=['paper_id']).paper_references.apply(lambda x: list(map(int, ast.literal_eval(x)))[0])
print(metrics.mapk(ans1.tolist(), first_predictions, 1))
all_refs = []
for i, data in dblp_remain.paper_references.iteritems():
    all_refs.extend(ast.literal_eval(data))
all_refs = Counter(all_refs)
hot_1 = all_refs.most_common(1)[0][0]

# todo 從dblp_remain找對應paper的references
plt.bar('Hot', metrics.mapk(y.tolist(), hot, 1))
plt.bar('RS', metrics.mapk(y.tolist(), first_predictions, 1))
plt.ylabel('score')
plt.title('MAP@1')
plt.show()


# TODO rolling的方式讓NN去學習embedding, for loop分年fit
# for i in range(8):
#     model.fit()


