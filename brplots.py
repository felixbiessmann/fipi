from classifier import Classifier
import seaborn as sns
from bokeh import mpl
from bokeh.plotting import output_file, show
import pandas as pd
from operator import itemgetter
import numpy as np

colors = ["purple","red","green","yellow","gray","blue"]

partyFiles = [
    ('Linke','41223_2013.csv'),
    ('SPD','41320_2013.csv'),
    ('Gruene','41113_2013.csv'),
    ('FDP','41420_2013.csv'),
    ('CDU',"41521_2013.csv"),
    ('AfD',"41953_2013.csv")
    ]

domains = [
    'External Relations',
    'Freedom and Democracy',
    'Political System',
    'Economy',
    'Welfare and Quality of Life'
    ]

def plotAll(folder = "data/manifesto/", nSamples = 100):

    clf = Classifier(train=False)

    predictions = []
    for party,fn in partyFiles:
        print("Getting texts of party %s"%party)
        data = pd.read_csv(folder+fn)
        nsamples = min(nSamples,len(data))
        predict = lambda x: groupPredictions(clf.predict(x))
        partyPredictions = data['content'].sample(nsamples).apply(predict).apply(pd.Series)
        partyPredictions['party'] = party
        predictions.append(partyPredictions)

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

def groupPredictions(prediction):
    domainTuples = [(x['label'],x['prediction']) for x in prediction['domain']]
    rightPrediction = [(x['label'],x['prediction']) for x in prediction['leftright']]
    result = {k:v for k,v in domainTuples + rightPrediction}
    return result
