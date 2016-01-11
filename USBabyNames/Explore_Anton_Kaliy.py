
### Notice
# This is NOT my work !!
# This is copy from https://www.kaggle.com/kapliy/d/kaggle/us-baby-names/exploring-us-baby-names

import warnings
warnings.filterwarnings("ignore",message="axes.color_cycle is deprecated")
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import sqlite3
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

# explore sqlite contents
con = sqlite3.connect('./data/database.sqlite')
cursor = con.cursor()
cursor.execute("SELECT NAME FROM SQLITE_MASTER WHERE TYPE='table';")
print(cursor.fetchall())

# helper method to load data
def load(what='NatinalNames'):
    assert what in ('NationalNames', 'StateNames')
    cols = ['Name', 'Year', 'Gender','Count']
    if what == 'StateNames':
        cols.append('State')
    df = pd.read_sql_query("SELECT {} from {}".format(','.join(cols),what),con)
    return df
df = load(what='NationalNames')
print df.head(5)

glob_freq = (df.groupby('Name')
                .agg({'Count': 'sum'})
                .sort('Count',ascending=False))
glob_freq[['Count']].head(10).plot(kind='bar')

# A random sample of unpopular names
print glob_freq.query('Count<=10').sample(10,random_state=2)

# visualize post WW2 baby boom
population = df[['Year','Count']].groupby('Year').sum()
population.plot()

# "Jackie" peaks during Kennedy presidency (thanks to the very popular first lady, Jackie Kennedy)
df.query('Name=="Jackie"')[['Year','Count']].groupby('Year').sum().plot()

# Are more male babies born?
tmp = df.groupby(['Gender','Year']).agg({'Count':'sum'}).reset_index()
male = (tmp.query("Gender == 'M'")
            .set_index("Year").sort_index()
            .rename(columns={'Count':'Male'}))
female = (tmp.query("Gender == 'F'")
            .set_index("Year").sort_index()
            .rename(columns={'Count':'Female'}))
join = male[['Male']].join(female[['Female']], how='outer')
join['Male Excess'] = join['Male'] - join['Female']
join.plot()

# Common names that are shared between girls and boys
tmp = df.groupby(['Gender', 'Name']).agg({'Count': 'sum'}).reset_index()
male = (tmp.query("Gender == 'M'")
        .set_index("Name")
        .rename(columns={'Count': 'Male'}))
female = (tmp.query("Gender == 'F'")
          .set_index("Name")
          .rename(columns={'Count': 'Female'}))
join = male[['Male']].join(female[['Female']], how='inner')
join['Frequency'] = join['Male'] + join['Female']
join['FemalePct'] = join['Female'] / join['Frequency'] * 100.0
join['MalePct'] = join['Male'] / join['Frequency'] * 100.0
print (join[['Frequency', 'FemalePct', 'MalePct']]
    .query('(FemalePct > 10) & (MalePct > 10)')
    .sort('Frequency',ascending=False)
    .head(10))

# state-by-state data
df2 = load(what='StateNames')
print df2.head(5)

# Evolution of baby births in the 10 largest states
tmp = df2.groupby(['Year','State']).agg({'Count':'sum'}).reset_index()
largest_states = (tmp.groupby('State')
                    .agg({'Count': 'sum'})
                    .sort('Count',ascending=False)
                    .index[:10].tolist())
tmp.pivot(index='Year',columns='State',values='Count')[largest_states].plot()

def most_similar_states(state,year=2014,top=3):
    """
    Returns the list of states where babies have the most similar names to a given state

    Details
    ============
    The state-by-state data is first converted to an TF-IDF matrix:
       https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    Then, cosine similarity is computed betweent the given state
    and every other state:
       https://en.wikipedia.org/wiki/Cosine_similarity
    Finally, the list of distances is sorted, and the states with the
    largest similarity are returned along with their cosine similarity metric.

    Arguments
    =============
    state      :   input state that will be compared with all other states
    year       :   (2014) use baby names from this year
    top        :   (3) return this many most-similar states

    """
    # compute a matrix where rows = states, columns = unique baby names(dictionary)
    features = pd.pivot_table(df2.query('Year=={}'.format(year)),
                              values='Count',index='State',columns='Name',aggfunc=np.sum)
    all_states = features.index.tolist()

    if state not in all_states:
        raise ValueError('Unknown state: {}'.format(state))

    idx = all_states.index(state)
    model = TfidfTransformer(use_idf=True,smooth_idf=True,norm='l2')
    tf = model.fit_transform(features.fillna(0))
    sims = pd.DataFrame.from_records(cosine_similarity(tf[idx:idx+1],tf),columns=all_states).T
    return (sims
            .rename(columns={0: 'similarity'})
            .sort('similarity', ascending=False)
            .head(top+1).tail(top) # remove first entry, since self-similarity is always 1
            )

most_similar_states(state='TX',year=2014,top=4)

most_similar_states(state='MA', year=2010, top=4)


plt.show()