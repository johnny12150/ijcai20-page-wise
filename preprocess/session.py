import pandas as pd
import ast
from tqdm import tqdm

dblp_2011 = pd.read_csv('dblp_2011.csv')
dblp_fill = pd.read_csv('dblp_fill.csv')
dblp = pd.read_pickle('dblp_2011up_venue_renamed.pkl')
# change year type from float to str
dblp['year'] = dblp['year'].astype(str)
# all related paper
dblp_all = dblp.append([dblp_2011, dblp_fill], ignore_index=True)
dblp_all['year'] = dblp_all['year'].astype(str)

tqdm.pandas()
# record userid and paperid
authors_paper = []
for i, data in tqdm(dblp.iterrows(), total=dblp.shape[0]):
    authors = ast.literal_eval(data.authors)  # get authors list
    for d in authors:
        d['paper_id'] = str(data.id)
    authors_paper.extend(authors)

# conver dict to df
dblp_author_paper = pd.DataFrame(authors_paper)
# 統計所有作者的歷史論文
authors = dblp_author_paper.groupby(['id', 'name'])['paper_id'].agg(','.join).reset_index()
print(dblp.sort_values(by=['year']).loc[:, ['id', 'year']].head()) # sort dblp by year
# 移除只有一篇paper的作者
authors[authors['paper_id'].map(lambda x: len(x.split(','))) > 1].shape

# todo: select paper wrote by the author and sort it by it's publication date
# for i, data in tqdm(authors.iterrows(), total=authors.shape[0]):
for i, data in authors.iterrows():
    public_paper = data['paper_id'].split(',')  # get the paper id list
    if len(public_paper) > 1:
        # select paper in dblp to find it's publication date
        print(dblp_all[dblp_all['id'].isin(public_paper)].sort_values(by=['year'])['id'].astype(str).tolist() == public_paper)
        # authors.loc[i, ] = dblp_all[dblp_all['id'].isin(public_paper)].sort_values(by=['year'])['id'].astype(str).tolist()
        break

# todo: split to basket set based on author

# todo: split basket set by conference and years
