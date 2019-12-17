import pickle
import pandas as pd
import ast

dblp_all = pickle.load(open('dblp_time_step.pkl', 'rb')).drop(columns=['doi', 'issue', 'page_end', 'page_start', 'publisher', 'volume'])
few_data = dblp_all.dropna(subset=['indexed_abstract','venue_id'])
few_data = few_data.loc[(few_data['venue_id'].astype(int) < 10) & (few_data['year'].astype(int) >= 2011)]

assert len(few_data.loc[(few_data['venue_id'].astype(int) > 10)]) == 0, 'venue num out range!'
assert len(few_data.loc[(few_data['year'].astype(int) < 2011)]) == 0, 'year num out range!'

# few_data.head()

refs = []  # 所有被引用過的
refs.extend(few_data['id'].astype(str))
for i, ref in few_data.dropna(subset=['references']).references.iteritems():
    refs.extend(ast.literal_eval(ref))

# 取獨一無二
refs = list(set(refs))
paper_node = len(refs)

refs = list(map(float, refs))
# 保留在reference裡的
dblp_all = dblp_all[dblp_all['id'].isin(refs)]

assert len(dblp_all) == paper_node, 'paper num out range!'

dblp_all.reset_index(drop=True, inplace=True)
dblp_all.reset_index(level=0, inplace=True)
dblp_all.rename(columns={"index": "new_papr_id"}, inplace=True)

dblp_all.to_csv('preprocess/edge/dblp_venue_top10.csv', index=False)
