import glob
import operator
import os
import codecs
import json
import pdb
import scipy as sp
from classifier import Classifier
from party_classifier import PartyClassifier
from sklearn import metrics, cross_validation
from sklearn.feature_extraction.text import HashingVectorizer,CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from itertools import chain
from sklearn.cross_validation import StratifiedKFold
import pandas as pd
from scipy import random,unique,vstack,hstack,arange,array,mean,corrcoef
import pickle

DATA_PATH = os.environ.get('DATA_PATH', 'data')
TXT_DIR = os.path.join(DATA_PATH, 'txt')
OUT_DIR = os.path.join(DATA_PATH, 'out')

MINLEN_SPEECH = 1000
MINLEN = 100

partyManifestoMap = {
    'gruene':41113,
    'cducsu':41521,
    'fdp':41420,
    'spd':41320,
    'afd':41953,
    'linke':41223,
    'pirates':41952
}

bundestagParties = {
    17:['gruene','cducsu','spd','fdp','linke'],
    18:['gruene','cducsu','spd','linke']
}

bundestagGovernment = {
    17:{'government':['cducsu','fdp'],'opposition':['gruene','spd','linke']},
    18:{'government':['cducsu','spd'],'opposition':['gruene','linke']}
}

bundestagSeats = {
    17:{'gruene':68,'cducsu':239,'spd':146,'fdp':93,'linke':76},
    18:{'gruene':63,'cducsu':311,'spd':193,'linke':64}
}

def nullPrediction(parties=['linke','gruene','spd','cducsu']):
    return dict([(k, 1.0/len(parties)) for k in parties])

def get_raw_text_bundestag_timeseries(folder="data/out", legislationPeriod=17):
    '''
    Loads raw text and labels from bundestag sessions
    (Preprocessed with https://github.com/bundestag/plpr-scraper, saved in json)
    '''
    data = []
    labels = []
    timestamp = []
    for f in glob.glob(folder+'/%d*.json'%legislationPeriod):
        speechdata,speechlabels=[],[]
        speeches = json.load(open(f))
        for speech in speeches:
            if speech['type']=='speech' and \
            speech['speaker_party'] is not None and \
            speech['speaker_party'] in bundestagParties[legislationPeriod] and \
            len(speech['text']) > MINLEN_SPEECH:
                speechdata.append(speech['text'])
                speechlabels.append(speech['speaker_party'])
        if len(speechdata) > 0:
            df = pd.DataFrame(zip(speechlabels,speechdata),columns=['labels','text']).groupby('labels').sum()
            labels.extend(df.index.tolist())
            data.extend(df.values.flatten().tolist())
            timestamp.extend([speech['filename']]*len(df))
    return data,labels,timestamp

def get_raw_text_bundestag(folder="data/out", legislationPeriod=17):
    '''
    Loads raw text and labels from bundestag sessions
    (Preprocessed with https://github.com/bundestag/plpr-scraper, saved in json)
    '''
    data = []
    labels = []
    for f in glob.glob(folder+'/%d*.json'%legislationPeriod):
        speeches = json.load(open(f))
        for speech in speeches:
           if speech['type']=='speech' and \
            speech['speaker_party'] is not None and \
            speech['speaker_party'] in bundestagParties[legislationPeriod] and \
            len(speech['text']) > MINLEN_SPEECH:
                data.append(speech['text'])
                labels.append(speech['speaker_party'])
    return data,labels

def run_experiments():
    for legis in [17,18]:
        #classify_speeches_binary_manifesto(legis);
        classify_speeches_binary_parliament(legis);
        #classify_speeches_party_manifesto(legis);
        classify_speeches_party_parliament(legis);
        #get_sentiments(legis)
        #get_word_correlations(legis)

def get_sentiments(legis=17):
    trainData, trainLabels = get_raw_text_bundestag(legislationPeriod=legis)
    se = SentimentClassifier()
    sentiments = {p:[] for p in bundestagParties[legis]}
    for item in zip(trainData,trainLabels):
        sentiments[item[1]].append(se.predict(item[0]))
    sortedParties = sorted(sentiments,key=bundestagSeats[legis].get)
    sortedSentiments = [sentiments[d] for d in sortedParties]
    gov = bundestagGovernment[legis]['government']
    sortedPartiesLabels = [p +" (%d)"%bundestagSeats[legis][p]+ "\n" + '[Government]' if p in gov else p+" (%d)"%bundestagSeats[legis][p]+"\n[Opposition]"for p in sortedParties]
    import pylab
    pylab.figure(figsize=(1.5*len(sortedParties),4))
    #pylab.boxplot(sortedSentiments,labels=sortedParties)
    col = {'gruene':'green','linke':'purple','spd':'red','cducsu':'black','fdp':'yellow'}
    pylab.bar(0.1+arange(len(sortedSentiments)),[mean(s) for s in sortedSentiments],color=[col[p] for p in sortedParties])
    pylab.grid('on')
    pylab.ylabel('Average Text Sentiment')
    pylab.xticks(0.5 + arange(len(sortedSentiments)), sortedPartiesLabels)
    pylab.xlim([0,len(sortedSentiments)+.01])
    font = {'family' : 'normal', 'size'   : 12}
    pylab.rc('font', **font)
    pylab.savefig(OUT_DIR+'/party-sentiments-%d.pdf'%legis,bbox_inches='tight')
    # correlation between sentiment and government membership
    member = hstack([vstack([sentiments[p],[1.0] * len(sentiments[p])]) if p in gov else vstack([sentiments[p],[-1.0] * len(sentiments[p])]) for p in sentiments.keys()])
    corMember = corrcoef(member[0,:],member[1,:])
    print("Corr sentiment vs government membership: %0.2f"%corMember[0,1])
    meanSentiment,govMember =zip(*[(mean(sentiments[p]),1.0) if p in gov else (mean(sentiments[p]), -1.0) for p in sentiments.keys()])
    print("Corr Mean Sentiment vs gov membership: %0.2f"%corrcoef(meanSentiment,govMember)[0,1])
    # correlation between sentiment and party seats
    seats = hstack([vstack([sentiments[p],[bundestagSeats[legis][p]] * len(sentiments[p])]) for p in sentiments.keys()])
    corSeats = corrcoef(seats[0,:],seats[1,:])
    print("Corr sentiment vs seats: %0.2f"%corSeats[0,1])
    # mean of parties
    meanSentiment,seats=zip(*[(mean(sentiments[p]),bundestagSeats[legis][p]) for p in sentiments.keys()])
    print("Corr Mean Sentiment vs seats: %0.2f"%corrcoef(meanSentiment,seats)[0,1])

def get_stops(legislationPeriod=17,includenames=True):
    # generic stopwords
    stopwords = codecs.open("data/stopwords.txt", "r", "utf-8").readlines()[7:]
    stops = [x.lower().strip() for x in stopwords]
    names = []
    if includenames:
        # names of abgeordnete
        abgeordnete = codecs.open('data/abgeordnete-%d.txt'%legislationPeriod,encoding='utf-8').readlines()
        names = [y.strip().lower().split(' ')[0] for x in abgeordnete for y in x.split(',')]
    return unique(stops + names).tolist()

def get_word_correlations(legis=17):
    trainData, trainLabels = get_raw_text_bundestag(legislationPeriod=legis)
    stops = get_stops()
    bow = TfidfVectorizer(max_df=0.1).fit(trainData)
    X = bow.transform(trainData)
    wordidx2word = dict(zip(bow.vocabulary_.values(),bow.vocabulary_.keys()))
    wordCors = {}
    for pa in bundestagParties[legis]:
        party = [1.0 if x==pa else -1.0 for x in trainLabels]
        partyLabel = vstack(party)
        print("Calculating correlations for %s"%pa)
        #co = cosine_similarity(X.T,partyLabel.T - mean(partyLabel))
        #import pdb;pdb.set_trace()
        co = array([corrcoef(X[:,wo].todense().flatten(),partyLabel.flatten())[0,1] for wo in range(X.shape[-1])])
        wordCors[pa] = dict([(wordidx2word[x],co[x]) for x in co.nonzero()[0] if wordidx2word[x] not in stops])
    json.dump(dict(wordCors),open(OUT_DIR+'/wordCors-%d.json'%legis,'wb'))

def list_top_words(legis=17,topwhat=10):
    import pylab
    fn = OUT_DIR+'/wordCors-%d.json'%legis
    cors = json.loads(open(fn).read())
    colors = {'linke':'purple','gruene':'green','spd':'red','cducsu':'black','fdp':'yellow'}
    cors = {key:sorted(cors[key].items(), key=operator.itemgetter(1)) for key in cors.keys()}
    stops = get_stops(legislationPeriod=legis)
    
    for party in cors.keys():
        pylab.figure(figsize=(3,10))
        c = [x for x in cors[party] if x[0] not in stops]
        tmp = c[:topwhat] + [('...',0)] +  c[-topwhat:]
        words = [x[0] for x in tmp]
        wordcors = [x[1] for x in tmp]
        pylab.barh(arange(len(words)),wordcors,color=colors[party])
        pylab.yticks(arange(len(words)),words)
        pylab.ylim(-.2,len(words))
        xl = max(abs(array(wordcors))) + .01
        pylab.xticks([-xl,0,xl])
        pylab.title(party)
        pylab.xlabel('Correlation')
        font = {'family' : 'normal', 'size'   : 16}
        pylab.rc('font', **font)
        pylab.savefig(OUT_DIR+'/party_word_correlations-%s-%d.pdf'%(party,legis),bbox_inches='tight')

def newsFileTuple(nf):
    return [(x['text'],x['source']) for x in json.load(open(nf))]

def get_raw_text_fb(folder="data/parteien-auf-fb", suffix=".json.gz"):
    '''
    Loads raw text and labels from news article files
    (Downloaded using newsreader.get_daily_news())
    '''
    files = glob.glob(folder+"/*"+suffix)
    return zip(*chain(*filter(None,map(fbFileTuple,files))))

def get_raw_text_fb_balanced_words(nWords=1000,folder="data/parteien-auf-fb", suffix=".json.gz"):
    data,labels = get_raw_text_fb(folder,suffix)
    newData, newLabels = [], []
    uparties, upcounts = sp.unique(labels,return_counts=True)
    for p in uparties:
        alltext = " ".join([x[0] for x in zip(data,labels) if x[1]==p])
        newTexts = map(lambda x: " ".join(x),zip(*[iter(alltext.split(" "))]*nWords))
        newData.extend(newTexts)
        newLabels.extend([p]*len(newTexts))
    return newData,newLabels

def fbFileTuple(fbf):
    f = [ff[1] for ff in fbTraverserTop(fbf)]
    return zip(f, [fbf.split("/")[-1].split(".")[0]] * len(f))

def get_raw_text_news(folder="data", prefix="news-2016"):
    '''
    Loads raw text and labels from news article files
    (Downloaded using newsreader.get_daily_news())
    '''
    files = glob.glob(folder+"/"+prefix+"*")
    return zip(*chain(*filter(None,map(newsFileTuple,files))))

def get_raw_text_manifesto(folder="data", legislationPeriod=17):
    '''
    Loads raw text and labels from manifestoproject csv files
    (Downloaded from https://visuals.manifesto-project.wzb.eu)
    '''
    #lines = open("all-german-manifesto-texts.csv").readlines()
    #return zip(*[(", ".join(l.split(",")[:-2]),int(l.split(",")[-2])) for l in lines[1:]])
    files = glob.glob(folder+"/[0-9]*_[0-9]*.csv")
    return zip(*chain(*filter(None,map(csv2ManifestoCodeTuple,files))))

def get_raw_text_manifesto_whole_country(fn):
    '''
    Loads raw text and labels from manifestoproject csv files
    (Downloaded from https://visuals.manifesto-project.wzb.eu)
    '''
    return zip(*chain(*filter(None,map(csv2ManifestoCodeTuple,[fn]))))
    

def csv2ManifestoCodeTuple(f):
    df = pd.read_csv(f,quotechar="\"")[['content','cmp_code']].dropna()
    return zip(df['content'].tolist(),df['cmp_code'].map(int).tolist())

def get_raw_text(folder="data", legislationPeriod=18):
    '''
    Loads raw text and labels from manifestoproject csv files
    (Downloaded from https://visuals.manifesto-project.wzb.eu)
    '''
    parties = bundestagParties[legislationPeriod]
    partyIds = [str(partyManifestoMap[p]) for p in parties]
    year = '2013'
    if legislationPeriod==17:year='2009'
    files = glob.glob(folder+"/[0-9]*_%s.csv"%year)
    files = filter(lambda x: x.split('/')[-1].split('_')[0] in partyIds,files)
    return zip(*chain(*filter(None,map(csv2DataTuple,files))))

def get_raw_text_topics_all_parties(folder="data", legislationPeriod=18):
    '''
    Loads raw text and labels from manifestoproject csv files
    (Downloaded from https://visuals.manifesto-project.wzb.eu)
    '''
    parties = bundestagParties[legislationPeriod]
    partyIds = [str(partyManifestoMap[p]) for p in parties]
    year = '2013'
    if legislationPeriod==17:year='2009'
    files = glob.glob(folder+"/[0-9]*_%s.csv"%year)
    files = filter(lambda x: x.split('/')[-1].split('_')[0] in partyIds,files)
    return zip(*chain(*filter(None,map(csv2DataTupleTopicsAllParties,files))))

def get_raw_text_topics(folder="data", legislationPeriod=18):
    '''
    Loads raw text and labels from manifestoproject csv files
    (Downloaded from https://visuals.manifesto-project.wzb.eu)
    '''
    parties = bundestagParties[legislationPeriod]
    partyIds = [str(partyManifestoMap[p]) for p in parties]
    year = '2013'
    if legislationPeriod==17:year='2009'
    files = glob.glob(folder+"/[0-9]*_%s.csv"%year)
    files = filter(lambda x: x.split('/')[-1].split('_')[0] in partyIds,files)
    return zip(*chain(*filter(None,map(csv2DataTupleTopics,files))))

def csv2DataTuple(f):
    '''
    Extracts list of tuples of (text,label) for each manifestoproject file
    '''
    df = pd.read_csv(f)
    df['content'] = df['content'].astype('str')
    partyId = f.split('/')[-1].split('_')[0]
    party = [k for (k,v) in partyManifestoMap.items() if str(v) == partyId]
    return zip(df['content'].tolist(), party * len(df))

def csv2DataTupleTopicsAllParties(f):
    '''
    Extracts list of tuples of (text,label) for each manifestoproject file
    '''
    df = pd.read_csv(f)
    df['content'] = df['content'].astype('str')
    df['topic'] = ((df['cmp_code'].dropna()) / 100).apply(lambda x: int(x))
    df = df.ix[df['topic'] > 0].dropna()
    return zip(df['content'].tolist(), df['topic'].tolist())

def csv2DataTupleTopics(f):
    '''
    Extracts list of tuples of (text,label) for each manifestoproject file
    '''
    df = pd.read_csv(f)
    df['content'] = df['content'].astype('str')
    df['topic'] = ((df['cmp_code'].dropna()) / 100).apply(lambda x: int(x))
    partyId = f.split('/')[-1].split('_')[0]
    party = [k for (k,v) in partyManifestoMap.items() if str(v) == partyId]
    contentByTopic = df.groupby('topic')['content'].sum()
    return zip(contentByTopic.tolist(), [party[0] + ":%d"%topic for topic in contentByTopic.index.tolist()])

def news_experiment(trainData,trainLabels,evalData,evalLabels, folds=2, idstr='default'):
    stops = get_stops()
    text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf',LogisticRegression(class_weight='balanced'))])
    parameters = {'vect__ngram_range': [(1,2)],\
           #'vect__stop_words':(None,stops),\
           #'tfidf__use_idf': (True,False),\
           'clf__C': (10.**sp.arange(-2,5,1.)).tolist(),
           'vect__max_df':[.1, .5, .75],
        }
    saveId = idstr+"-"+randid()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainData, trainLabels, test_size=0.1, random_state=0)
    # optimize hyperparams on training set
    gs_clf = GridSearchCV(text_clf, parameters, cv=StratifiedKFold(y_train, folds), n_jobs=-1,verbose=1)
    # train on training set
    best_clf = gs_clf.fit(X_train, y_train).best_estimator_
    # test on test set
    test_clf = text_clf.set_params(**best_clf.get_params()).fit(X_train,y_train)
    predictedTest = test_clf.predict(X_test)
    # dump report on training held out data with CV
    report = "*** Training Set (CVd) ***\n" + metrics.classification_report(y_test, predictedTest)
    labelsStr = [str(x) for x in test_clf.steps[-1][1].classes_]
    report += '\nConfusion Matrix (rows=true, cols=predicted)\n'+', '.join(labelsStr)+'\n'
    for line in metrics.confusion_matrix(y_test, predictedTest).tolist(): report += str(line)+"\n" 
    # train again on entire training set
    final_clf = text_clf.set_params(**best_clf.get_params()).fit(trainData, trainLabels)
    # dump report on training held out data with CV
    labels = {'welt': {'cducsu':.743+.029, 'fdp':.086},
    'faz': {'cducsu':.743, 'fdp':.057},
    'sz': {'spd':.686,'fdp':.057,'gruene':.085, 'cducsu':.0290}}
    # transform human judgements into vectors so we can compare with classifier output
    y = [[labels[evl][l] if l in labels[evl] else 0.0 for l in labelsStr] if evl in labels else [0]*len(labelsStr) for evl in evalLabels]
    humanJudgements = sp.array(y)
    predictedEval = final_clf.predict_proba(evalData)
    C = corrcoef(predictedEval,humanJudgements)
    corrs = sp.diag(C[predictedEval.shape[0]:,:humanJudgements.shape[0]])
    # correlations of human newspaper jugdement and party predictions per news paper
    npp = [[corrs[idx] for idx,evl in enumerate(evalLabels) if evl == l] for l in labels.keys()]
    perc = [25,50,75]
    corrsPercentiles = dict([(np,[sp.percentile(npp[labels.keys().index(np)],p) for p in perc]) for np in labels.keys()])
    
    print("\n".join([x[0]+":\tperc "+"\t".join(["%d:%0.2f"%y for y in zip(perc,x[1])]) for x in corrsPercentiles.items()]))
    import pdb;pdb.set_trace()

def optimize_hyperparams(trainData,trainLabels,evalData,evalLabels, folds=2, idstr='default'):
    #stops = get_stops()
    text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf',LogisticRegression(class_weight='balanced'))])
    parameters = {'vect__ngram_range': [(1,2)],\
           #'vect__stop_words':(None,stops),\
           #'tfidf__use_idf': (True,False),\
           'clf__C': (10.**sp.arange(-2,5,1)).tolist(),
           'vect__max_df':[.1,.2,.5,7],
        }
    saveId = idstr+"-"+randid()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainData, trainLabels, test_size=0.1, random_state=0)
    # optimize hyperparams on training set
    gs_clf = GridSearchCV(text_clf, parameters, cv=StratifiedKFold(y_train, folds), n_jobs=-1,verbose=1)
    # train on training set
    best_clf = gs_clf.fit(X_train, y_train).best_estimator_
    # test on test set
    test_clf = text_clf.set_params(**best_clf.get_params()).fit(X_train,y_train)
    predictedTest = test_clf.predict(X_test)
    # dump report on training held out data with CV
    report = "*** Training Set (CVd) ***\n" + metrics.classification_report(y_test, predictedTest)
    labelsStr = [str(x) for x in test_clf.steps[-1][1].classes_]
    report += '\nConfusion Matrix (rows=true, cols=predicted)\n'+', '.join(labelsStr)+'\n'
    for line in metrics.confusion_matrix(y_test, predictedTest).tolist(): report += str(line)+"\n" 
    report += "Accuracy: %0.2f\n"%metrics.accuracy_score(y_test,predictedTest)
    # train again on entire training set
    final_clf = text_clf.set_params(**best_clf.get_params()).fit(trainData, trainLabels)
    # dump classifier to pickle
    pickle.dump(final_clf,open(OUT_DIR+'/pipeline-'+saveId+'.pickle','wb'))
    # test on evaluation data ste
    predictedEval = final_clf.predict(evalData)
    report += "*** Evaluation Set ***\n" + metrics.classification_report(evalLabels, predictedEval)
    labelsStr = [str(x) for x in final_clf.steps[-1][1].classes_]
    report += '\nConfusion Matrix (rows=true, cols=predicted)\n'+', '.join(labelsStr)+'\n'
    for line in metrics.confusion_matrix(evalLabels, predictedEval).tolist(): report += str(line)+"\n" 
    report += "Accuracy: %0.2f\n"%metrics.accuracy_score(evalLabels,predictedEval)
    # dump report
    open(OUT_DIR+'/report-'+saveId+'.txt','wb').write(report.encode('utf-8'))
    return report

def randid(N=10):
    return "".join(map(lambda x: str(x),random.randint(0,9,N).tolist()))

def classify_manifesto_codes_topics():
    data, labels = get_raw_text_manifesto_whole_country('data/UK/United_Kingdom.csv')
    labels = [round(l/100) for l in labels]
    trainData, evalData, trainLabels, evalLabels = cross_validation.train_test_split(data, labels, test_size=0.1, random_state=0)
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="manifesto_codes_topics")

def classify_manifesto_codes():
    data, labels = get_raw_text_manifesto_whole_country('data/UK/United_Kingdom.csv')
    #data, labels = get_raw_text_manifesto()
    trainData, evalData, trainLabels, evalLabels = cross_validation.train_test_split(data, labels, test_size=0.1, random_state=0)
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="manifesto_codes")

def classify_speeches_binary_manifesto(legislationPeriod = 18):
    evalDataParty, evalLabelsParty = get_raw_text_bundestag(legislationPeriod=legislationPeriod)
    trainDataParty, trainLabelsParty = get_raw_text(legislationPeriod=legislationPeriod)
    gov = bundestagGovernment[legislationPeriod]['government']
    trainData, trainLabels = zip(*[(x[0],'government') if x[1] in gov else (x[0],'opposition') for x in zip(trainDataParty,trainLabelsParty)])
    evalData, evalLabels = zip(*[(x[0],'government') if x[1] in gov else (x[0],'opposition') for x in zip(evalDataParty,evalLabelsParty)])
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="manifesto-train-gov-%d"%legislationPeriod)


def classify_speeches_binary_parliament(legislationPeriod = 18):
    trainDataParty, trainLabelsParty = get_raw_text_bundestag(legislationPeriod=legislationPeriod)
    evalDataParty, evalLabelsParty = get_raw_text(legislationPeriod=legislationPeriod)
    gov = bundestagGovernment[legislationPeriod]['government']
    trainData, trainLabels = zip(*[(x[0],'government') if x[1] in gov else (x[0],'opposition') for x in zip(trainDataParty,trainLabelsParty)])
    evalData, evalLabels = zip(*[(x[0],'government') if x[1] in gov else (x[0],'opposition') for x in zip(evalDataParty,evalLabelsParty)])
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="parliament-train-gov-%d"%legislationPeriod)


def classify_manifesto_topics_all_parties():
    trainData, trainLabels = get_raw_text_topics_all_parties(legislationPeriod=17)
    evalData, evalLabels = get_raw_text_topics_all_parties(legislationPeriod=18)
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="topics-all-parties")

def classify_speeches_parliament_topics(legislationPeriod = 18):
    trainData, trainLabels = get_raw_text_bundestag(legislationPeriod=legislationPeriod)
    evalData, evalLabels = get_raw_text_topics(legislationPeriod=legislationPeriod)
    optimize_hyperparams(trainData,trainLabels, evalData, [e.split(":")[0] for e in evalLabels],idstr="parliament-train-gov-topics-%d"%legislationPeriod)

def classify_speeches_binary_parliament_topics(legislationPeriod = 18):
    trainDataParty, trainLabelsParty = get_raw_text_bundestag(legislationPeriod=legislationPeriod)
    evalDataParty, evalLabelsParty = get_raw_text_topics(legislationPeriod=legislationPeriod)
    gov = bundestagGovernment[legislationPeriod]['government']
    trainData, trainLabels = zip(*[(x[0],'government') if x[1] in gov else (x[0],'opposition') for x in zip(trainDataParty,trainLabelsParty)])
    evalData, evalLabels = zip(*[(x[0],'government') if x[1].split(":")[0] in gov else (x[0],'opposition') for x in zip(evalDataParty,evalLabelsParty)])
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="parliament-train-gov-binary-topics-%d"%legislationPeriod)

def classify_speeches_party_parliament(legislationPeriod = 18):
    trainData, trainLabels = get_raw_text_bundestag(legislationPeriod=legislationPeriod)
    evalData, evalLabels = get_raw_text(legislationPeriod=legislationPeriod)
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="parliament-train-%d"%legislationPeriod)

def classify_speeches_fb(legislationPeriod = 18, nsamples=50):
    evalData, evalLabels = get_raw_text_fb_balanced_words(nWords=1000)
    #evalData, evalLabels = get_raw_text_fb()
    uparties, upcounts = sp.unique(evalLabels,return_counts=True)
    dd = zip(evalData,evalLabels)
    sp.random.shuffle(dd)
    hh = [[x for x in dd if x[1]==up] for up in uparties]
    evalData,evalLabels = zip(*chain(*[h[:nsamples] for h in hh]))
    trainData, trainLabels = get_raw_text_bundestag(legislationPeriod=legislationPeriod)
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="fb-test-%d"%legislationPeriod)

def classify_speeches_fb_binary(legislationPeriod = 18, nsamples=50):
    evalData, evalLabels = get_raw_text_fb_balanced_words(nWords=1000)
    #evalData, evalLabels = get_raw_text_fb()
    uparties, upcounts = sp.unique(evalLabels,return_counts=True)
    dd = zip(evalData,evalLabels)
    sp.random.shuffle(dd)
    hh = [[x for x in dd if x[1]==up] for up in uparties]
    evalData,evalLabels = zip(*chain(*[h[:nsamples] for h in hh]))
    trainData, trainLabels = get_raw_text_bundestag(legislationPeriod=legislationPeriod)
    gov = bundestagGovernment[legislationPeriod]['government']
    trainData, trainLabels = zip(*[(x[0],'government') if x[1] in gov else (x[0],'opposition') for x in zip(trainData,trainLabels)])
    evalData, evalLabels = zip(*[(x[0],'government') if x[1] in gov else (x[0],'opposition') for x in zip(evalData,evalLabels)])
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="fb-binary-%d"%legislationPeriod)

def classify_speeches_party_manifesto(legislationPeriod = 18):
    evalData, evalLabels = get_raw_text_bundestag(legislationPeriod=legislationPeriod)
    trainData, trainLabels = get_raw_text(legislationPeriod=legislationPeriod)
    optimize_hyperparams(trainData,trainLabels, evalData, evalLabels,idstr="manifesto-train-%d"%legislationPeriod)

def classify_speeches_party_news(legislationPeriod = 17):
    evalData, evalLabels = get_raw_text_news()

    trainData, trainLabels = get_raw_text(legislationPeriod=legislationPeriod)
    news_experiment(trainData,trainLabels, evalData, evalLabels,idstr="manifesto-train-%d"%legislationPeriod)

def party_similarity(text,bow,partyVecs):
    cosine_similarity(partyVecs,bow.transform([text]))

def plot_similarity_sessions_party_manifesto(legislationPeriod = 18):
    evalData, evalLabels, timestamp = get_raw_text_bundestag_timeseries(legislationPeriod=legislationPeriod)
    trainData, trainLabels = get_raw_text_topics(legislationPeriod=legislationPeriod)
    stops = get_stops()
    text_preprocessing = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),stop_words=stops)),('tfidf', TfidfTransformer())])
    # train bow vectorizer
    bow = text_preprocessing.fit(evalData + list(trainData))
    mf = bow.transform(trainData)
    df = pd.DataFrame(zip(evalData, evalLabels, timestamp),columns=['text','labels','timestamp'])
    df['sims'] = [cosine_similarity(mf, bow.transform([t[1][0]])).flatten() for t in df.iterrows()]
    sessions = df.groupby(['timestamp','labels'])
    parties = df['labels'].unique().tolist()
    partySimTimeseries = {k:[] for k in parties}
    for sess,idcs in sessions.groups.items():
        partySimTimeseries[sess[1]].append(df.ix[idcs]['sims'].tolist())

    import pylab as pl
    pl.figure(figsize=(7,10))
    pl.ion()
    for p in parties:
        pl.clf()
        pl.imshow(sp.vstack(partySimTimeseries[p]).T,aspect='auto',interpolation='nearest')
        pl.yticks(range(len(trainLabels)),trainLabels)
        pl.xlabel("#session")
        pl.title("Similarity session BOW vector %s with manifesto BOW"%p)
        pl.savefig(OUT_DIR + "/timeseries-%s-legislationPeriod-%d.pdf"%(p,legislationPeriod))
        
    

def classify_speeches_party(legislationPeriod = 18):
    from party_classifier import PartyClassifier
    #optimize_hyperparams_party(legislationPeriod)
    pclf = PartyClassifier(train=True)
    predictedParty = []
    trueParty = []
    for f in glob.glob(OUT_DIR+'/17*.json'):
        speeches = json.load(open(f))
        print("processing %d speeches in %s"%(len(speeches),f))
        for speech in speeches:
            if speech['type']=='speech' and \
            speech['speaker_party'] is not None and \
            speech['speaker_party'] in bundestagParties[legislationPeriod] and \
            len(speech['text']) > MINLEN:
                prediction = pclf.predict(speech['text'])
                predictedParty.append(sp.argmax(prediction.values()))
                trueParty.append(prediction.keys().index(speech['speaker_party']))
    report = metrics.classification_report(trueParty, \
                predictedParty,target_names=prediction.keys())
    report += '\nConfusion Matrix (rows=true, cols=predicted)\n'+', '.join(prediction.keys())+'\n'
    for line in metrics.confusion_matrix(trueParty, predictedParty).tolist():
        report += str(line)+"\n" 
    open(OUT_DIR+'/report','wb').write(report)
    import pdb;pdb.set_trace()
    report += "Accuracy: %d" + accuracy_score(trueParty,predictedParty)
    return report

def plot_leftright():
    import pylab as pl
    predTs = {
        'gruene':[],
        'cducsu':[],
        'spd':[],
        'fdp':[],
        'linke':[]
    }
    for f in glob.glob(OUT_DIR+'/17*predictions.json'):
        pred = json.load(open(f))
        for party in pred.keys():
            if len(pred[party]['leftright'])>0:
                predTs[party].append(pred[party]['leftright']['right'])
        
    pl.figure(figsize=(20,10))
    for party in pred.keys():
        predTs[party] = sp.stack(predTs[party],1)
        pl.clf()
        pl.plot(predTs[party].T)
        pl.savefig(OUT_DIR+'/partyTimeseries_'+party+'.pdf')
            

def get_party_predictions():
    pred_init = {'leftright':{},'manifestocode':{}}
    pred = {
            'gruene': {'leftright':{},'manifestocode':{}},
            'cducsu': {'leftright':{},'manifestocode':{}},
            'spd': {'leftright':{},'manifestocode':{}},
            'fdp': {'leftright':{},'manifestocode':{}},
            'linke': {'leftright':{},'manifestocode':{}}
            }
    for f in glob.glob(OUT_DIR+'/17*.json'):
        speeches = json.load(open(f))
        print("processing %d speeches in %s"%(len(speeches),f))
        for speech in speeches:
            if speech['type']=='speech' and \
            speech['speaker_party'] is not None and \
            pred.has_key(speech['speaker_party']):
                prediction = clf.predict(speech['text'])
                for prediction_type in pred_init.keys():
                    for pr in prediction[prediction_type]:
                        k = pr['label']
                        v = pr['prediction']
                        if not pred[speech['speaker_party']][prediction_type].has_key(k): 
                            pred[speech['speaker_party']][prediction_type][k] = [v]
                        else:
                            pred[speech['speaker_party']][prediction_type][k].append(v)
    for party in pred.keys():
        for prediction_type in pred[party].keys():
            for pr in pred[party][prediction_type].keys():
                pred[party][prediction_type][pr] = sp.percentile(pred[party][prediction_type][pr],[5, 25, 50, 75, 95]).tolist()
    json.dump(pred,open(OUT_DIR+'/predictions.json','wb'))

def get_party_predictions_daily():
    pred_init = {'leftright':{},'manifestocode':{}}
    for f in glob.glob(OUT_DIR+'/17*.json'):
        pred = {
                'gruene': {'leftright':{},'manifestocode':{}},
                'cducsu': {'leftright':{},'manifestocode':{}},
                'spd': {'leftright':{},'manifestocode':{}},
                'fdp': {'leftright':{},'manifestocode':{}},
                'linke': {'leftright':{},'manifestocode':{}}
                }
        speeches = json.load(open(f))
        print("processing %d speeches in %s"%(len(speeches),f))
        for speech in speeches:
            if speech['type']=='speech' and \
            speech['speaker_party'] is not None and \
            pred.has_key(speech['speaker_party']):
                prediction = clf.predict(speech['text'])
                for prediction_type in pred_init.keys():
                    for pr in prediction[prediction_type]:
                        k = pr['label']
                        v = pr['prediction']
                        if not pred[speech['speaker_party']][prediction_type].has_key(k): 
                            pred[speech['speaker_party']][prediction_type][k] = [v]
                        else:
                            pred[speech['speaker_party']][prediction_type][k].append(v)
        for party in pred.keys():
            for prediction_type in pred[party].keys():
                for pr in pred[party][prediction_type].keys():
                    pred[party][prediction_type][pr] = sp.percentile(pred[party][prediction_type][pr],[5, 25, 50, 75, 95]).tolist()
        json.dump(pred,open(f.replace(".json","-predictions.json"),'wb'))
    


def classify_all_speeches():
    for f in glob.glob(OUT_DIR+'/*.json'):
        classify_speeches(f)

def classify_speeches(f):
    data = json.load(open(f))
    data = [v for v in data if v['type'] == 'speech']
    for speech in data:
        speech['predictions'] = clf.predict(speech['text'])
    json.dump(data,open(f.replace('.json','-with-classification.json'),'wb'))

def get_all_bundestags_data():
    fetch_protokolle()
    for filename in os.listdir(TXT_DIR):
        parse_transcript_json(os.path.join(TXT_DIR, filename))

def parse_transcript_json(filename):

    wp, session = file_metadata(filename)
    with open(filename, 'rb') as fh:
        text = clean_text(fh.read())

    data = []

    base_data = {
        'filename': filename,
        'sitzung': session,
        'wahlperiode': wp
    }
    print("Loading transcript: %s/%.3d, from %s" % (wp, session, filename))
    seq = 0
    parser = SpeechParser(text.split('\n'))

    for contrib in parser:
        contrib.update(base_data)
        contrib['sequence'] = seq
        contrib['speaker_cleaned'] = clean_name(contrib['speaker'])
        contrib['speaker_fp'] = fingerprint(contrib['speaker_cleaned'])
        contrib['speaker_party'] = search_party_names(contrib['speaker'])
        seq += 1
        data.insert(0,contrib)

    jsonfile = os.path.basename(filename).replace('.txt', '.json')
    json.dump(data,open(os.path.join(OUT_DIR,jsonfile),'wb'))

 
