import pandas as pd
import ast
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate

bert_title = 768
bert_abstract = 768
# output dim (就是graph embedding後的dim)
emb_dim = 100

dblp_top50_conf = pd.read_pickle('preprocess/edge/paper_2011up_venue50.pkl')
pa = pd.read_pickle('preprocess/edge/paper_author.pkl')
pv = pd.read_pickle('preprocess/edge/paper_venue.pkl')
# print(dblp_top50_conf.year.value_counts())  # 各年度幾篇
paperId_compare = dblp_top50_conf.loc[:, ['id', 'new_papr_id', 'venue_id']]  # id與new_id的對照表

# print(dblp_top50_conf.new_papr_id.max())  # 找paper的max id

# authors = []
# for i, data in dblp_top50_conf.authors.iteritems():
#     for j in ast.literal_eval(data):
#         authors.append(j['id'])
# print(len(authors))  # 透過算作者有幾個找max id, 395952個

# paper_new_id + author_id 後即為venue_id
vId_start = dblp_top50_conf.new_papr_id.max() + 395952
# vId, pId 對照表 (從新paperId找其venueId與authorId)
# dblp_top50_conf['new_aId'] = dblp_top50_conf['new_aId'] + 395952
# dblp_top50_conf['new_vId'] = dblp_top50_conf['venue_id'] + vId_start

# select 2018以前全部當train
train2017 = dblp_top50_conf.loc[dblp_top50_conf.year < 2018, ['new_papr_id', 'title', 'new_vId']]


# https://stackoverflow.com/questions/41888085/how-to-implement-word2vec-cbow-in-keras-with-shared-embedding-layer-and-negative
# 用一個network去逼近embedding
paperId_emb = Input(shape=(emb_dim,))  # graph embedding的結果
authorId_emb = Input(shape=(emb_dim,))  # graph embedding的結果
title_emb = Input(shape=(bert_title,))  # BERT的結果
abstract_emb = Input(shape=(bert_abstract,))  # BERT的結果

out_emb = Dense(emb_dim, activation='sigmoid')(concatenate([paperId_emb, authorId_emb]))
model = Model(input=[paperId_emb, authorId_emb], output=out_emb)

# 最後一層用sigmoid/ linear輸出100個units, loss用mae硬做
model.compile(optimizer='rmsprop', loss='mae')

# 找出跟NN predict出最相似的embedding當作要推薦cite的論文

# 先用2018前全部 train推薦系統, 2019的test推薦效果
train_history = model.fit()

test_history = model.predict()

# TODO rolling的方式讓NN去學習embedding, for loop分年fit
for i in range(8):
    model.fit()
