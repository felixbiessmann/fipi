from classifier import Classifier, label2domain, manifestolabels
# from sklearn.feature_extraction.text import
import seaborn as sns
from bokeh import mpl
from bokeh.plotting import output_file, show
import pandas as pd
from operator import itemgetter
import numpy as np
import re
from scipy.spatial.distance import cdist

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
    'Welfare and Quality of Life'
    ]

# def compute_most_distant_statements()

def read_md(fn):
    '''
    Reads manifesto from md file
    '''
    md_text = open(fn).read()
    # texts = map(lambda x: re.sub("\s+"," ",x),re.split('(#+ [\d\.]+.*\n)',md_text))
    # texts = list(map(lambda x: re.sub("\s+"," ",x),re.split('(#+ [\d\.]+.*\n)',md_text)))
    texts = list(map(lambda x: re.sub("\s+"," ",x),re.split('#+',md_text)))
    return texts

def classify_br(folder, fn, party, clf):
    content = read_md(folder + fn)
    preds = clf.predictBatch(content)
    preds['max_manifesto'] = preds[list(manifestolabels().values())].idxmax(axis=1)
    preds['max_domain'] = preds[list(label2domain.keys())].idxmax(axis=1)
    preds['max_leftright'] = preds[['left', 'right']].idxmax(axis=1)
    preds['content'] = content
    preds['party'] = party
    return preds

def plotAll(folder = FOLDER):
    predictions = []
    colors = []
    clf = Classifier(train=True)
    for party, fn, color in partyFiles:
        predictions.append(classify_br(folder, fn, party, clf))
        colors.append(color)

    df = pd.concat(predictions)
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
