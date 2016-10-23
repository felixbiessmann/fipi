import pandas as pd
from scipy import corrcoef,array,mean
import os,re
from polyglot.text import Text

EUFN = "europarl-speeches.csv"
LANGREGEXP = "|".join(["sv","en","it","fr","es","de","nl","pl"])
#SEATS = {'EPP':221,'S&D':191,'ECR':70,'ALDE':67,'GUE/NGL':52,'Greens/EFA':50,'EFDD':48,'NI':52}
SEATS = {'EPP':221,'ECR':70,'ALDE':67,'GUE':52,'EFDD':48,'NI':52}

def compute_sentiment_text(text):
    sentiment = 0
    try:
        sentiment = mean([mean([w.polarity for w in s.words]) for s in text.sentences])
    except:
        pass
    return sentiment 

def compute_sentiment(fn=EUFN,download=False):
    df = pd.read_csv(fn)
    df = df[df.text.apply(type) == str]
    df.language = df.language.apply(lambda x: x[:2])
    #df = df[df.language.str.match(LANGREGEXP)]
    if download:
        [os.system("polyglot download sentiment2.%s"%lang) for lang in df.language.unique()]
    df['TextPoly'] = df.apply(lambda x:Text(x.text,hint_language_code=x.language),axis=1)
    df['Sentiment'] = df.apply(lambda x: compute_sentiment_text(x.TextPoly) if x.pol_group in SEATS.keys() else 0,axis=1)
    sentiment = {k:(df[df.pol_group==k].Sentiment.mean(),v) for k,v in SEATS.items()}   
    s,v = zip(*sentiment.values())
    print("Correlation number of seats vs sentiment %0.2f"%corrcoef(s,v)[0,1])



