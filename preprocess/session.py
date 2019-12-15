import pandas as pd
import ast
from tqdm import tqdm

dblp = pd.read_pickle('dblp_2011up_venue_renamed.pkl')

tqdm.pandas()
pbar = tqdm(total=dblp.shape[0])
# record userid and paperid
authors_paper = []
for i, data in dblp.iterrows():
    authors = ast.literal_eval(data.authors)  # get authors
    for j, au in enumerate(authors):
        # fixme: use df to increase performance
        # author paper dictionary
        if not any(d['author_name'] == au['name'] for d in authors_paper):
            authors_paper.append({'author_name': au['name'], 'author_id': au['id'], 'paper_id': [data.id]})
        else:
            for d in authors_paper:
                if d['author_id'] == au['id']:
                    d['paper_id'].append(data.id)

pbar.close()

# todo: split to basket set based on author

# todo: split basket set by conference and years
