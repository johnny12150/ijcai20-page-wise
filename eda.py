import pandas as pd
import ast

dblp = pd.read_pickle('dblp_2011up_venue_renamed.pkl')
# dblp_all = pd.read_pickle("F:/volume/19' summer vacation/0819/concat_df/dblp_papers_4100000.pkl")
dblp_all = pd.read_pickle('dblp_papers_v11.pkl')

print(dblp.info())

papers_authors = []
for i, data in dblp.authors.iteritems():
    # string to obj/ json-like
    # print(ast.literal_eval(data))
    # print(pd.DataFrame(ast.literal_eval(data)))
    papers_authors.extend(ast.literal_eval(data))

# 一個作者有幾篇
print(pd.DataFrame(papers_authors).name.value_counts())
print(len(pd.DataFrame(papers_authors).name.value_counts()))
# 超過10片的作者有幾個
print(sum(pd.DataFrame(papers_authors).name.value_counts() > 10))

fos = []
# fos
for i, data in dblp.dropna(subset=['fos']).fos.iteritems():
    fos.extend(ast.literal_eval(data))

print(pd.DataFrame(fos).name.value_counts())
print(sum(pd.DataFrame(fos).name.value_counts() > 3))

refs = []  # 所有被引用過的
for i, ref in dblp.dropna(subset=['references']).references.iteritems():
    refs.extend(ast.literal_eval(ref))

# 取獨一無二
refs = list(set(refs))
refs = list(map(float, refs))
# 保留在reference裡的
dblp_all = dblp_all[dblp_all['id'].isin(refs)]
# 找2011前的
dblp_2011 = dblp_all.loc[dblp_all['year'] < 2011]
dblp_2011.to_csv('dblp_2011.csv', index=False)

# 找2011以後, 但不在我們候選名單裡的
refill = list(set(refs) - set(dblp_2011['id'].values.tolist()) - set(dblp['id'].values.tolist()))
dblp_refill = dblp_all[dblp_all['id'].isin(refill)]
dblp_refill.to_csv('dblp_fill.csv', index=False)

# 不要期刊
# dblp.loc[dblp['doc_type'] == 'Conference']
