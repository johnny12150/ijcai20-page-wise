import pandas as pd
import ast
from tqdm import tqdm

dblp = pd.read_pickle('dblp_2011up_venue_renamed.pkl')

tqdm.pandas()
# record userid and paperid
authors_paper = []
for i, data in tqdm(dblp.iterrows(), total=dblp.shape[0]):
    authors = ast.literal_eval(data.authors)  # get authors list
    for d in authors:
        d['paper_id'] = str(data.id)
    # fixme: use df to increase performance
    authors_paper.extend(authors)

# conver dict to df
dblp_author_paper = pd.DataFrame(authors_paper)
# 統計所有作者的歷史論文
authors = dblp_author_paper.groupby(['id', 'name'])['paper_id'].agg(','.join).reset_index()

# todo: split to basket set based on author

# todo: split basket set by conference and years
