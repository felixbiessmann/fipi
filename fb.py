import json,gzip
from itertools import chain
import cPickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation,metrics
import scipy as sp
import re

DDIR = "/Users/felix/Code/Python/fipi/data/parteien-auf-fb"

dat = [DDIR + x + ".json.gz" for x in ['afd','npd','pegida']]

urlPat = r'(http://.*\.html)'

def getUrlsAndUsers(post): 
    urls = []
    if post['message']: 
        foundUrl = re.search(urlPat,post['message'])
        if foundUrl:
            name = ''
            if post.has_key('name'):
                name += post['name']
            import pdb;pdb.set_trace()
            urls.append((post['created'],post['id']+": "+name,foundUrl.groups(0)[0]))
    if len(post['comments'])>0: 
        urls += list(chain(*map(getUrls,post['comments'])))
    return filter(lambda x: len(x)>0,urls)

def getUrls(post): 
    urls = []
    if post['message']: 
        foundUrl = re.search(urlPat,post['message'])
        if foundUrl:
            name = ''
            if post.has_key('name'):
                name += post['name']
            urls.append((post['created'],post['id']+": "+name,foundUrl.groups(0)[0]))
    if len(post['comments'])>0: 
        urls += list(chain(*map(getUrls,post['comments'])))
    return filter(lambda x: len(x)>0,urls)

def getText(post): 
    text = []
    if post['message'] and len(post['message'])>0: 
        text.append((post['created'][:10],post['message']))
    if len(post['comments'])>0: 
        text = text + list(chain(*map(getText,post['comments'])))
    return text

def getTextTop(post): 
    text = []
    if post['message'] and len(post['message'])>0: 
        text.append((post['created'][:10],post['message']))
    return text

def train_bow(d = dat):
    vect = CountVectorizer().fit(map(lambda x: x[1], chain(*map(fbTraverser, d))))
    cPickle.dump(vect,open(DDIR+"vectorizer.pickle","wb"))

def get_daily_bow_single(data, vect):
    daily_bow = {}
    for day,text in fbTraverser(data):
        if daily_bow.has_key(day):
            daily_bow[day] += vect.transform([text]) 
        else: daily_bow[day] = vect.transform([text])
    return daily_bow

def get_daily_bow(data = dat):
    vect = cPickle.load(open(DDIR+"vectorizer.pickle"))
    daily_bow = {k:get_daily_bow_single(k,vect) for k in data}
    cPickle.dump(daily_bow,open(DDIR+"daily_bow.pickle","wb"))

def top_group_words(topwhat=10):
    vect = cPickle.load(open(DDIR+"vectorizer.pickle"))
    bow = cPickle.load(open(DDIR+"daily_bow.pickle"))
    X,y = zip(*chain(*[zip(v.values(),len(v)*[k.split("/")[-1].split(".")[0]]) for k,v in bow.items()]))
    X = sp.sparse.vstack(X)
    wordidx2word = dict(zip(vect.vocabulary_.values(),vect.vocabulary_.keys()))
    topwords = {}
    for gr in sp.unique(y):
        yy = sp.array([1.0 if x==gr else -1.0 for x in y])
        covs = sp.array(X.T.dot(sp.array(yy))).flatten()
        idx = covs.argsort()
        topwords[gr] = {wordidx2word[i]:covs[i] for i in idx[:topwhat].tolist() + idx[-topwhat:].tolist()}
    return topwords

def classify_texts():
    bow = cPickle.load(open(DDIR+"daily_bow.pickle"))
    X,y = zip(*chain(*[zip(v.values(),len(v)*[k.split("/")[-1].split(".")[0]]) for k,v in bow.items()]))
    X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.3)
    parameters = {'clf__C':(10.**sp.arange(-4,5)).tolist()}
    text_clf = Pipeline([('clf',LogisticRegression(class_weight='balanced'))])
    gs_clf = GridSearchCV(text_clf,parameters,cv=2)
    best_clf = gs_clf.fit(sp.sparse.vstack(X_train),y_train).best_estimator_
    y_hat = best_clf.predict(sp.sparse.vstack(X_test))
    print(metrics.classification_report(y_test,y_hat))

class fbTraverser(object):
    def __init__(self, fn):
        self.fn = fn

    def __iter__(self):
        with gzip.open(self.fn) as fh:
            for line in fh:
                try:
                    for text in getText(json.loads(line)):
                        yield text
                except:
                    pass

class fbTraverserTop(object):
    def __init__(self, fn):
        self.fn = fn

    def __iter__(self):
        with gzip.open(self.fn) as fh:
            for line in fh:
                try:
                    for text in getTextTop(json.loads(line)):
                        yield text
                except:
                    pass


class fbTraverserUrl(object):
    def __init__(self, fn):
        self.fn = fn

    def __iter__(self):
        with gzip.open(self.fn) as fh:
            for line in fh:
                yield getUrls(json.loads(line, strict=False))
                    

class fbTraverserUrlUsers(object):
    def __init__(self, fn):
        self.fn = fn

    def __iter__(self):
        with gzip.open(self.fn) as fh:
            for line in fh:
                yield getUrlsAndUsers(json.loads(line, strict=False))
                    



