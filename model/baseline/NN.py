import pandas as pd
import numpy as np
import pickle
from keras.models import Model
from keras.layers import Input, Dense, concatenate

bert_title = 768
bert_abstract = 768
# output dim (就是graph embedding後的dim)
emb_dim = 100

dblp_top50_conf = pd.read_pickle('preprocess/edge/paper_venue.pkl')
pa = pd.read_pickle('preprocess/edge/p_a_delete_author.pkl')
# ab = pd.read_pickle('preprocess/edge/abstracts.pkl')  # FIXME 編碼不對, 且空值未做處理

# FIXME train的時候只考慮第一作者 ?
# https://stackoverflow.com/questions/20067636/pandas-dataframe-get-first-row-of-each-group
dblp_top50_conf['new_first_aId'] = pa.groupby('new_papr_id').first()['new_author_id']  # 取每篇的第一作者
dblp_top50_conf.dropna(subset=['new_first_aId'], inplace=True)  # 移除沒有作者的paper

# TODO 輸出成一個刪減後的paper csv

# select 2018以前全部當train
train2017 = dblp_top50_conf.loc[dblp_top50_conf.time_step < 2018, ['new_papr_id', 'new_venue_id', 'new_first_aId']]
test2018 = dblp_top50_conf.loc[dblp_top50_conf.time_step >= 2018, ['new_papr_id', 'new_venue_id', 'new_first_aId']]

# https://stackoverflow.com/questions/41888085/how-to-implement-word2vec-cbow-in-keras-with-shared-embedding-layer-and-negative
# 用一個network去逼近embedding
paperId_emb = Input(shape=(emb_dim,))  # graph embedding的結果
authorId_emb = Input(shape=(emb_dim,))  # graph embedding的結果
title_emb = Input(shape=(bert_title,))  # BERT的結果
abstract_emb = Input(shape=(bert_abstract,))  # BERT的結果

# all_in_one = Input(shape=(emb_dim*2+bert_title+bert_abstract,))
all_in_one = Input(shape=(emb_dim*2,))

# out_emb = Dense(emb_dim, activation='sigmoid')(concatenate([paperId_emb, authorId_emb, title_emb, abstract_emb]))
d1 = Dense(512, activation='linear')(all_in_one)
d2 = Dense(400, activation='linear')(d1)
d3 = Dense(256, activation='linear')(d2)
out_emb = Dense(emb_dim, activation='sigmoid')(d3)
model = Model(input=all_in_one, output=out_emb)

# 最後一層用sigmoid/ linear輸出100個units, loss用mae硬做
model.compile(optimizer='rmsprop', loss='mae')

# 找出跟NN predict出最相似的embedding當作要推薦cite的論文

node_id = np.load('preprocess/edge/node_id.npy').astype(int)
emb_2017 = np.load('preprocess/edge/deep_walk_vec.npy')
# sort both np at the same time
# sorter = node_id.argsort()
# emb_2017_sorted = emb_2017[sorter]

# line embedding
with open('preprocess/edge/line_1and2.pkl', 'rb') as file:
    line_emb = pickle.load(file)


# todo data generator, 或是直接concat起來送進keras
def embedding_loader(emb_data, file_len, batch_size=32):
    while True:
        batch_x = []
        batch_y = []
        batch_paths = np.random.choice(a=file_len, size=batch_size)
        for batch_i in batch_paths:
            # TODO find node id 在emb_data的index

            # 找出該paper的所有資訊
            emb_p = emb_data[str(batch_i)]  # paper emb
            # emb_p = emb_data[batch_i]
            # 根據新paper id 找出 aId, vId
            vId, aId = dblp_top50_conf.loc[batch_i, ['new_venue_id', 'new_first_aId']]
            # emb = np.hstack(emb_data[[vId, aId], :])  # np style
            emb1 = emb_data[str(vId)]
            emb2 = emb_data[str(int(aId))]
            emb = np.concatenate((emb1, emb2), axis=None)
            batch_x += [emb]
            batch_y += [emb_p]
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        print(batch_x.shape)
        yield batch_x, batch_y


batch = 32
# 先用2018前全部 train推薦系統, 2019的test推薦效果
# train_history = model.fit_generator(embedding_loader(emb_2017, train2017.new_papr_id.values), epochs=1, steps_per_epoch=train2017.shape[0] / batch, verbose=1)
train_history = model.fit_generator(embedding_loader(line_emb, train2017.new_papr_id.values), epochs=1, steps_per_epoch=train2017.shape[0] / batch, verbose=1)

# test_history = model.predict()

# TODO rolling的方式讓NN去學習embedding, for loop分年fit
# for i in range(8):
#     model.fit()
