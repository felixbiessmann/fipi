import gzip,itertools,glob,re
import numpy as np
from classifier import Classifier
from scipy import double,zeros,corrcoef,vstack
from scipy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

DDIR = "fb-partyposts"

def load_sentiment(negative='SentiWS_v1.8c/SentiWS_v1.8c_Negative.txt',\
        positive='SentiWS_v1.8c/SentiWS_v1.8c_Positive.txt'):
    words = dict()
    for line in open(negative).readlines():
        parts = line.strip('\n').lower().split('\t')
        words[parts[0].split('|')[0]] = double(parts[1])
        if len(parts)>2:
            for inflection in parts[2].strip('\n').split(','):
                words[inflection] = double(parts[1])

    for line in open(positive).readlines():
        parts = line.strip('\n').lower().split('\t')
        words[parts[0].split('|')[0]] = double(parts[1])
        if len(parts)>2:
            for inflection in parts[2].strip('\n').split(','):
                words[inflection] = double(parts[1])

    return words

def predictRight(clf,text):
    p = clf.predict(text)
    right = [x['prediction'] for x in p['leftright'] if x['label']=='right']
    return right

def predictSentiment(text,sentiWords):
    s = 0.0
    for w in text.split(' '):
        ww = re.sub('\W+','', w.lower())
        if ww in sentiWords: s += sentiWords[ww]
    return s

def getPartyTextsDF(ddir=DDIR):
    fns=glob.glob(ddir+"/*.tsv")
    allData = {fn:processFacebookData(fn) for fn in fns}
    clf = Classifier()
    sentiWords = load_sentiment()
    df = pd.DataFrame(list(itertools.chain(*allData.values())),columns=['party','likes','text'])
    df['sentiment'] = df['text'].apply(lambda x: predictSentiment(x,sentiWords))
    return df

def getTextByPartyOR(df,words,party):
    mask = np.logical_or.reduce([df['text'].str.contains(w) for w in words])
    return df[mask][df['party']==party].sort('sentiment')

def getTextByPartyAND(df,words,party):
    mask = np.logical_and.reduce([df['party']==party]+[df['text'].str.contains(w) for w in words])
    return df[mask].sort('sentiment')

def processParties(ddir=DDIR,topWhat=1000):
    fns=glob.glob(ddir+"/*.tsv")
    allData = {fn:processFacebookData(fn) for fn in fns}
    # train one BoW model first
    texts = itertools.chain(*[[x[-1] for x in d] for d in allData.values()])
    sentiWords = load_sentiment()
    stops = map(lambda x:x.lower().strip(),open('stopwords.txt').readlines()[6:])
    texts = itertools.chain([" ".join(sentiWords.keys())],texts)
    bow = TfidfVectorizer(min_df=2,max_df=.05,stop_words=stops).fit(texts)
    wordIdx2Word = {v:k for k,v in bow.vocabulary_.items()}
    sentiVec = bow.transform([" ".join(sentiWords.keys()).lower()])
    for key in sentiWords:
        if key in bow.vocabulary_:
            sentiVec[0,bow.vocabulary_[key]] = sentiWords[key]
    result = {}
    for fn in fns:
        result[fn] = {}
        bowVecs = bow.transform([x[-1] for x in allData[fn]])
        bowVecsBinary = bowVecs
        bowVecsBinary.data = bowVecsBinary.data / bowVecsBinary.data
        sentimentPost = bowVecsBinary.dot(sentiVec.T)

        sentLikeCorr = corrcoef(sentimentPost.toarray().flatten(),[int(x[1]) for x in allData[fn]])[0,1]
        print("fn: %s, correlation Likes vs Post sentiment: %0.2f"%(fn.split("/")[-1],sentLikeCorr))
        wordSentimentContext = bowVecsBinary.T.dot(sentimentPost).toarray().flatten()
        result[fn]['topPosSentimentContext'] = [wordIdx2Word[idx] for idx in wordSentimentContext.argsort()[-1::-1][:topWhat]]
        result[fn]['topNegSentimentContext'] = [wordIdx2Word[idx] for idx in wordSentimentContext.argsort()[:topWhat]]
        result[fn]['sentimentPost'] = sentimentPost
        result[fn]['wordSentimentContext'] = wordSentimentContext

    for fni in fns:
        meanOthers = vstack([result[fnj]['wordSentimentContext']/norm(result[fnj]['wordSentimentContext']) for fnj in fns if fnj != fni]).mean(axis=0)
        dif = result[fni]['wordSentimentContext'] - meanOthers
        result[fni]['topPosSentimentContextDiff'] = [wordIdx2Word[idx] for idx in dif.argsort()[-1::-1][:topWhat]]
        result[fni]['topNegSentimentContextDiff'] = [wordIdx2Word[idx] for idx in dif.argsort()[:topWhat]]
        setOthers = set(itertools.chain(*[result[fnj]['topPosSentimentContext'] for fnj in fns if fnj != fni]))
        result[fni]['distinctPositive'] = set(result[fni]['topPosSentimentContext']).difference(setOthers.union(set(sentiWords.keys())))
        setOthers = set(itertools.chain(*[result[fnj]['topNegSentimentContext'] for fnj in fns if fnj != fni]))
        result[fni]['distinctNegative'] = set(result[fni]['topNegSentimentContext']).difference(setOthers.union(set(sentiWords.keys())))
        from pylab import savefig,hist,figure
        figure()
        bins = [-1.,-.5,-.2,-.1,0.,.1,.2]
        sent = result[fni]['sentimentPost'].toarray()
        hist(sent / max(abs(sent)),bins)
        savefig(fni[:-4]+"-sentimentPostHistogram.pdf")

    return result,wordIdx2Word

def get_sentiment(text,bow):

    bow = TfidfVectorizer(min_df=0)
    X = bow.fit_transform([text])

    # map sentiment vector to bow space
    words = load_sentiment()
    sentiment_vec = zeros(X.shape[1])
    for key in words.keys():
        if key in bow.vocabulary_:
            sentiment_vec[bow.vocabulary_[key]] = words[key]

    # compute sentiments
    return X.dot(sentiment_vec)

def processFacebookData(fn):
    recs = []
    for post in open(fn).readlines():
        try:
            party,_,_,numberLikes,numberComments,\
                _,_,_,text = post.split("\t")
            recs.append([party,numberLikes,text.lower()])
        except:
            Warning("Could not process %s"%post)
            pass
    return recs
