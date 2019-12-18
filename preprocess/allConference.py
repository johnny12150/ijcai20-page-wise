import pickle
import ast
import pandas as pd
from tqdm import tqdm

dblp_all = pickle.load(open('dblp_time_step.pkl', 'rb')).drop(columns=['doi', 'issue', 'page_end', 'page_start', 'publisher', 'volume'])
few_data = dblp_all.dropna(subset=['indexed_abstract', 'venue_id'])
few_data = few_data.loc[(few_data['venue_id'].astype(int) < 51) & (few_data['year'].astype(int) >= 2011)]

print(few_data.shape)

# 整理出剩下paper中的所有references
refs = []
for i, ref in few_data.dropna(subset=['references']).references.iteritems():
    refs.extend(ast.literal_eval(ref))
refs = list(set(refs))
refs = list(map(float, refs))

tqdm.pandas()
# 計算每篇paper還會剩下多少edge
print(few_data[few_data['id'].isin(refs)].shape)

remain_papers = []
for i, data in few_data.dropna(subset=['references']).reset_index(drop=True).iterrows():
    paper_ref = {'paper_id': data.id, 'paper_references': [], 'references_amount': 0}
    for pa in ast.literal_eval(data.references):
        if few_data['id'].isin([int(pa)]).any():
            paper_ref['paper_references'].append(pa)
    paper_ref['references_amount'] = len(paper_ref['paper_references'])
    remain_papers.append(paper_ref)

print(pd.DataFrame(remain_papers).head())

pd.DataFrame(remain_papers).to_csv('preprocess/edge/dblp_remain_references.csv', index=False)

# save it
# few_data.rename(columns={"index": "new_papr_id"}, inplace=True)
# few_data.to_csv('preprocess/edge/dblp_venue_conferences.csv', index=False)
