import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt

dblp_2011 = pd.read_csv('dblp_2011.csv')
dblp_fill = pd.read_csv('dblp_fill.csv')
dblp = pd.read_pickle('dblp_2011up_venue_renamed.pkl')
# change year type from float to str
dblp['year'] = dblp['year'].astype(str)
# fill na reference with empty string
dblp.fillna(value={'references': ''}, inplace=True)
# all related paper
dblp_all = dblp.append([dblp_2011, dblp_fill], ignore_index=True)
dblp_all['year'] = dblp_all['year'].astype(str)

tqdm.pandas()
# record userid and paperid
authors_paper = []
# for i, data in tqdm(dblp.iterrows(), total=dblp.shape[0]):
for i, data in dblp.iterrows():
    authors = ast.literal_eval(data.authors)  # get authors list
    for d in authors:
        d['paper_id'] = str(data.id)
        d['venue_name'] = data.venue_name
        d['year'] = str(data.year)
        d['venue_id'] = str(data.venue_id)
    authors_paper.extend(authors)

# conver dict to df
dblp_author_paper = pd.DataFrame(authors_paper)
# 每年每個conference的session長度
year_conference = dblp_author_paper.groupby(['venue_name', 'year'])['id'].count()
year_conference_index = list(set(year_conference.index.get_level_values(0).tolist()))

# 根據每個conference畫他每年的session長度
for j in range(len(year_conference_index)):
    year_count = year_conference.loc[[year_conference_index[j]]].reset_index()
    year_range = range(2011, 2011+len(year_count))
    plt.plot(year_count.year.values, year_count.id.values)
    plt.title(year_conference_index[j])
    plt.savefig('./preprocess/plots/year_conference/'+ year_conference_index[j].replace(': ', '') +'.png')
    plt.close()
    if j ==5:
        break

# count total conference with years
print(len(year_conference.index.get_level_values(0).tolist()))  # around 852

# 統計所有作者的歷史論文, sort in group
# https://stackoverflow.com/questions/27842613/pandas-groupby-sort-within-groups
# TODO conference should also consider month
authors = dblp_author_paper.sort_values(by=['year']).groupby(['id', 'name'])['paper_id'].agg(','.join).reset_index()
# print(dblp.sort_values(by=['year']).loc[:, ['id', 'year']].head()) # sort dblp by year
# 移除只有一篇paper的作者
authors = authors[authors['paper_id'].map(lambda x: len(x.split(','))) > 1].reset_index(drop=True)


pbar = tqdm(total=authors.shape[0])
# TODO create session of baskets
author_sessions = []
for i, data in authors.iterrows():
    # find each paper's reference, replace paper id with refernce basket
    # fixme: 應該根據不同篇paper的reference建basket
    paper_basket = dblp.loc[dblp['id'].isin(data.paper_id.split(',')), 'references'].tolist()
    paper_basket_tmp = []
    for l, ba in enumerate(paper_basket):
        try:
            # 該作者每篇paper的所有references
            paper_basket_tmp.append(ast.literal_eval(ba))
        except:
            # avoid empty references
            continue
    author_sessions.append(paper_basket_tmp)
    pbar.update(1)
pbar.close()
del pbar

# padding sessions
padded_sessions = np.zeros([len(author_sessions), len(max(author_sessions, key = lambda x: len(x)))])
for j, se in enumerate(author_sessions):
    padded_sessions[j][0:len(se)] = se
    if j==5:
        break

# TODO rolling by years and conference

# todo: split to basket set based on author

# todo: split basket set by conference and years
