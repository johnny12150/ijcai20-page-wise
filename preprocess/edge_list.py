import pandas as pd
import ast
from tqdm import tqdm

# dblp = pd.read_pickle('dblp_2011up_venue_renamed.pkl')
# dblp.fillna(value={'references': ''}, inplace=True)

# 先只用top 10 conference
dblp = pd.read_csv('preprocess/edge/dblp_venue_top10.csv')
dblp.fillna(value={'references': '', 'venue_id': 0}, inplace=True)
print(dblp.isna().sum())

# FIXME 從 0開始 mapping新的venue, paper, author id

tqdm.pandas()

resultlist = []
for i, data in tqdm(dblp.iterrows(), total=dblp.shape[0], position=0, leave=True):
    # link paper to reference
    if data.references:  # avoid empty value
        refs = ast.literal_eval(data.references)
        for j, ref in enumerate(refs):
            # create tuple for citation
            paper_link = (data.id, ref)
            rev_paper_link = (ref, data.id)
            resultlist.append(paper_link)

    # link paper to venue
    if data.venue_id:
        # 根據年分來建立venue node
        paper_link = (data.id, str(int(data.year))+str(int(data.venue_id)))
        rev_paper_link = (str(int(data.year))+str(int(data.venue_id)), data.id)
        # avoid any duplicates
        if paper_link not in resultlist and rev_paper_link not in resultlist:
            resultlist.append(paper_link)

        # 主要的venue node (不考慮年分)
        paper_link = (data.id, int(data.venue_id))
        rev_paper_link = (int(data.venue_id), data.id)
        resultlist.append(paper_link)

    # link paper to author
    if data.authors:  # avoid empty value
        aus = ast.literal_eval(data.authors)
        for j, au in enumerate(aus):
            paper_link = (data.id, au['id'])
            rev_paper_link = (au['id'], data.id)
            resultlist.append(paper_link)


# 使用set來取看會不會比較快
# resultlist = list(set(resultlist))  # 缺少考慮 node i to j is the same as node j to i
# list(set([tuple(reversed(t)) for t in resultlist]))  # 直接inverse也會少考慮
# 考慮node i & node j 的inverse的關係
result = list(set([tuple(sorted(map(int, t))) for t in resultlist]))

# the above section may take around 35 mins to run

# write tuple to txt
with open('preprocess/edge/papers_edge_list.txt', 'w') as f:
    for t in result:
        line = ' '.join(str(x) for x in t)
        f.write(line + '\n')


