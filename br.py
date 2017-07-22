from classifier import Classifier, label2domain, manifestolabels
# from sklearn.feature_extraction.text import
import seaborn as sns
from bokeh import mpl
from bokeh.plotting import output_file, show
import pandas as pd
from operator import itemgetter
import numpy as np
import scipy as sp
import re
import random
from scipy.spatial.distance import cdist
from sklearn.feature_extraction.text import TfidfVectorizer

FOLDER = "/Users/felix/Dropbox/work/br/data/parteiprogramme/"

# colors = [,"red","yellow"]

partyFiles = [
    ('AfD',"afd.md", "blue"),
    ('CDU/CSU', "cducsu.md", "gray"),
    ('FDP', "fdp.md", "yellow"),
    ('SPD', "spd.md", "red"),
    ('Grüne', "diegruenen.md", "green"),
    ('Die Linke', "dielinke.md", "purple")
    ]

domains = [
    'External Relations',
    'Freedom and Democracy',
    'Political System',
    'Economy',
    'Welfare and Quality of Life',
    'Fabric of Society'
    ]

# def compute_most_distant_statements()

def clean_whitespace(txt): return re.sub("\s+"," ",txt)

def read_md(fn, min_len=100):
    '''
    Reads manifesto from md file
    '''
    # split_symbol = '[\.\!\?\;] '#
    split_symbol = '#+'
    md_text = open(fn).read()
    len_filter = lambda x: len(x) > min_len
    # texts = map(lambda x: re.sub("\s+"," ",x),re.split('(#+ [\d\.]+.*\n)',md_text))
    # texts = list(map(lambda x: re.sub("\s+"," ",x),re.split('(#+ [\d\.]+.*\n)',md_text)))
    texts = filter(len_filter, map(clean_whitespace, re.split(split_symbol,md_text)))
    return list(texts)

def classify_br(folder, fn, party, clf, max_txts=5000):
    content = read_md(folder + fn)
    if len(content) > max_txts:
        content = random.sample(content, max_txts)
    preds = clf.predictBatch(content)
    preds['max_manifesto'] = preds[list(manifestolabels().values())].idxmax(axis=1)
    preds['max_domain'] = preds[list(label2domain.keys())].idxmax(axis=1)
    preds['max_leftright'] = preds[['left', 'right']].idxmax(axis=1)
    preds['content'] = content
    preds['party'] = party
    return preds

def compute_most_distant_statements_per_topic(preds, n_most_distant=5, folder=FOLDER):
    # compute nearest texts per party
    tf = TfidfVectorizer().fit(preds.content)
    preds['tf_idf'] = preds.content.apply(lambda x: tf.transform([x]))
    most_distant_statements = []
    for domain in domains:
        for party in [x[0] for x in partyFiles]:
            this_party = (preds.party == party) & (preds.max_domain == domain)
            other_parties = (preds.party != party) & (preds.max_domain == domain)
            partyVecs = sp.sparse.vstack(preds[this_party]['tf_idf'])
            partyTexts = preds[this_party]['content']
            otherVec = sp.sparse.vstack(preds[other_parties]['tf_idf']).sum(axis=0)
            dists = sp.array(abs(partyVecs - otherVec).sum(axis=1)).flatten()
            most_distant = partyTexts[dists.argsort()[-n_most_distant:][-1::-1]]
            most_distant_statements.extend([(party, domain, m) for m in most_distant])
    most_distant_statements_df = pd.DataFrame(most_distant_statements, columns=['party', 'domain', 'most_distant_to_other_parties'])
    most_distant_statements_df = most_distant_statements_df.sort_values(by=['party','domain'])
    most_distant_statements_df.to_csv(FOLDER+'most_distant_statements_per_topic.csv',index=False)
    return most_distant_statements_df

def plotAll(folder = FOLDER):
    predictions = []
    colors = []
    clf = Classifier(train=False)
    for party, fn, color in partyFiles:
        predictions.append(classify_br(folder, fn, party, clf))
        colors.append(color)

    df = pd.concat(predictions)
    # compute most distant statements per topic, discard result as it's csv-dumped
    _ = compute_most_distant_statements_per_topic(df, folder=folder)

    sns.set_style("whitegrid")

    for domain in domains:
        idx = df[domains].apply(pd.Series.argmax,axis=1)==domain
        ax = sns.violinplot(x="right",y="party",
            data=df[idx][['right','party']], palette=sns.color_palette(colors),
            split=True,scale="count", inner="stick", saturation=0.5)
        ax.set_xlim([0,1])
        ax.set_xticks(np.arange(0,1,.1))
        ax.set_xlabel("p(rechts)")
        ax.set_ylabel("Partei")
        ax.set_title(domain)
        output_file(folder+"violinPlot-%s.html"%domain)

        show(mpl.to_bokeh())
    df.to_csv(FOLDER + "results.csv")

def groupPredictions(prediction):
    domainTuples = [(x['label'],x['prediction']) for x in prediction['domain']]
    rightPrediction = [(x['label'],x['prediction']) for x in prediction['leftright']]
    result = {k:v for k,v in domainTuples + rightPrediction}
    return result
