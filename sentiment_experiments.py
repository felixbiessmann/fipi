import pandas as pd
from scipy import corrcoef,array,mean,median
import os,re
from polyglot.text import Text
from sklearn.feature_extraction.text import TfidfVectorizer

EUFN = "europarl-speeches.csv"
SPFN = "europarl-speakers.csv"
LANGREGEXP = "|".join(["sv","en","it","fr","es","de","nl","pl"])
#SEATS = {'EPP':221,'S&D':191,'ECR':70,'ALDE':67,'GUE/NGL':52,'Greens/EFA':50,'EFDD':48,'NI':52}
SEATS = {   'epp':221,
            'sd':191,
            'ecr':70,
            'aldeadle':67,
            'eensefa':50,
            'engl':52,
            'efd':48}#,
            # 'NI':52}

def compute_sentiment_text(text):
    sentiment = 0
    try:
        sentiment = mean([w.polarity for w in text.words])
    except:
        Warning('Could not compute sentiment for %s'%text)
        pass
    return sentiment

def compute_ner(text):
    ner = []
    try:
        ner = ["%s:%s"%(entity.tag, entity) for entity in text.entities]
    except:
        Warning('Could not compute NER for %s'%text)
        pass
    return ner


def compute_sentiment(fn=EUFN,speaker_fn=SPFN,download=False):
    df = pd.merge(pd.read_csv(fn),pd.read_csv(speaker_fn),on='speaker_id')
    df = df[df.text.apply(type) == str]
    df.language = df.language.apply(lambda x: x[:2])

    #df = df[df.language.str.match(LANGREGEXP)]
    if download:
        [os.system("polyglot download sentiment2.%s"%lang) for lang in df.language.unique()]
    df['TextPoly'] = df.apply(lambda x:Text(x.text,hint_language_code=x.language),axis=1)
    # df['Sentiment'] = df.apply(lambda x: compute_sentiment_text(x.TextPoly) if x.curr_pol_group_abbr in SEATS.keys() else 0,axis=1)
    df['Sentiment'] = df.apply(lambda x: compute_sentiment_text(x.TextPoly),axis=1)
    sentiment = {k:(df[df.curr_pol_group_abbr==k].Sentiment.mean(),v) for k,v in SEATS.items()}
    s,v = zip(*sentiment.values())
    print("Correlation number of seats vs sentiment %0.2f"%corrcoef(s,v)[0,1])
    # df['topicText'] = df.apply(lambda x:Text(x.topic,hint_language_code=x.language),axis=1)
    # df['topicNER'] = df.apply(lambda x: compute_ner(x.topicText),axis=1)
    # NERS = unique(list(itertools.chain(*df['topicNER'].values)))
    # locNERS = [x for x in NERS if "LOC" in x]
    bow = TfidfVectorizer(min_df=5,max_df=.05,token_pattern="[a-z]+").fit(df.topic)
    wordIdx2Word = {v:k for k,v in bow.vocabulary_.items()}
    df['senti_bow_topics']=df.apply(lambda x:x.Sentiment * bow.transform([x.topic]),axis=1)
    topWhat = 100
    sentiment,topWords,flopWords = {},{},{}
    for p in SEATS.keys():
        sentiment[p] = df.senti_bow_topics[df.curr_pol_group_abbr==p].sum() / sum(df.curr_pol_group_abbr==p)
        wordRanks = sentiment[p].toarray().flatten().argsort()
        topWords[p] = [wordIdx2Word[widx] for widx in wordRanks[-topWhat:][::-1]]
        flopWords[p] = [wordIdx2Word[widx] for widx in wordRanks[:topWhat]]
    uTopWords,uFlopWords = {},{}
    for p in SEATS.keys():
        others = set(itertools.chain(*[topWords[k] for k in topWords.keys() if k is not p]))
        uTopWords[p] = set(topWords[p]).difference(others)
        others = set(itertools.chain(*[flopWords[k] for k in flopWords.keys() if k is not p]))
        uFlopWords[p] = set(flopWords[p]).difference(others)
    party,seats,numUF = zip(*[(k,SEATS[k],numberUniqueFlopwords[k]) for k in numberUniqueFlopwords.keys()])
    party,seats,numUT = zip(*[(k,SEATS[k],numberUniqueTopwords[k]) for k in numberUniqueTopwords.keys()])
    # figure()
    # plot(seats,numUF,'ro')
    # plot(seats,numUT,'bo')
    # pylab.legend(['neg','pos'])
    # [pylab.text(seats[p]-15,numUF[p]+1.5,party[p]) for p in range(len(party))]
    # [pylab.text(seats[p],numUT[p]+1.5,party[p]) for p in range(len(party))]
    # pylab.xlabel("#Seats")
    # pylab.ylabel("#unique words")
    # savefig("numberUniqueTopwords.pdf")
    # figure()
    # pylab.plot(v,s,'o')
    # [pylab.text(v[1],v[0],k) for k,v in sentiment.items()]
    # pylab.ylabel('Avg Sentiment')
    # pylab.xlabel('#Seats')
    # pylab.savefig("EUSentiment-summed.pdf")
