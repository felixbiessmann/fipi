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

def plotAllDomains(folder = "data/manifesto/"):
    for dom in domains:
        plotAll(domain=dom)

def plotAll(folder = "data/manifesto/", domain = "Economy"):
    clf = Classifier(train=False)
    predictions = []
    for party,fn in partyFiles:
        print("Getting texts of party %s"%party)
        data = pd.read_csv(folder+fn)
        nsamples = min(100,len(data))
        pred = getLeftRightForDomain(data.sample(nsamples).content,clf,domain)
        predictions += [(x[0],x[1],party) for x in pred]

    df = pd.DataFrame(predictions,columns=['rightPrediction','text','party'])
    sns.set_style("whitegrid")

    ax = sns.violinplot(x="rightPrediction",y="party",
                    data=df, palette=sns.color_palette(colors), split=True,
                    scale="count", inner="stick", saturation=0.4)
    ax.xlim([0,1])
    ax.xticks(np.arange(0,1,.1))
    ax.xlabel("p(rechts)")
    ax.ylabel("Partei")
    ax.title(domain)
    output_file(folder+"violinPlot-%s.html"%domain)

    show(mpl.to_bokeh())

def getLeftRightForDomain(texts,clf,domain="Economy"):
    result = []
    for text in texts:
        prediction = clf.predict(text)
        domainTuples = [(x['label'],x['prediction']) for x in prediction['domain']]
        topDomain = sorted(domainTuples,key=itemgetter(1))[-1][0]
        if topDomain == domain:
            rightPrediction = [x['prediction'] for x in prediction['leftright'] if x['label'] is 'right'][0]
            result.append((rightPrediction, text))
    return result
