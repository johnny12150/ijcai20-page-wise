import pandas
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate

# output dim (就是graph embedding後的dim)
emb_dim = 100

# 目前先random出一個embedding

# https://stackoverflow.com/questions/41888085/how-to-implement-word2vec-cbow-in-keras-with-shared-embedding-layer-and-negative
# 用一個network去逼近embedding
paperId = Input(shape=(1,))
paperId_emb = Embedding(output_dim=512, input_dim=10000, input_length=100)(paperId)

authorId = Input(shape=(1,))
authorId_emb = Embedding(output_dim=512, input_dim=10000, input_length=100)(authorId)

out_emb = Dense(emb_dim, activation='sigmoid')(concatenate([paperId_emb, authorId_emb]))
model = Model(input=[paperId, authorId], output=out_emb)

# 最後一層用sigmoid/ linear輸出100個units, loss可以嘗試cross-entropy
model.compile(optimizer='rmsprop', loss='cross_entropy')

# 找出跟NN predict出最相似的embedding當作要推薦cite的論文

# rolling的方式讓NN去學習embedding, for loop分年fit
for i in range(8):
    model.fit()
