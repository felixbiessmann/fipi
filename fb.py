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
import urllib
from bs4 import BeautifulSoup

DDIR = "/Users/felix/Code/Python/fipi/data/parteien-auf-fb"

dat = [DDIR + x + ".json.gz" for x in ['afd','npd','pegida']]

urlPat = r'(http://.*\.html)'

def get_text_from_url(url):
    html = urllib.urlopen(url).read()
    soup = BeautifulSoup(html,"lxml")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if len(chunk.split(" "))>1)
    return text.encode('utf-8')

def getLikes(post):
    likes = []
    if post.has_key("likedBy"):
        likes = [post['likedBy']]   
    if len(post['comments'])>0: 
        likes += map(getLikes,post['comments'])
    return chain(*likes)
    

def getContent(post):
    text,likes,urlText,name,url,content = "",[],"","","",{}
    if post['message']: 
        text = post['message']
        foundUrl = re.search(urlPat,post['message'])
        if foundUrl:
            url = foundUrl.groups(0)[0]
            urlText = get_text_from_url(url)
        if post.has_key('name'): 
            name = post['name']
        if post.has_key("likedBy"):
            likes = post['likedBy']    
        content = {
            "url":url,
            "urlText":urlText,
            "text":text,
            "likes":likes,
            "date":post['created'],
            "id":post['id'] + ": "+name 
            }
    if len(post['comments'])>0: 
        content['likes'] = list(getLikes(post)) + likes
    return content


def getContentFlatOld(post): 
    urls = []
    if post['message']: 
        foundUrl = re.search(urlPat,post['message'])
        if foundUrl:
            name = ''
            if post.has_key('name'):
                name += post['name']
            if post.has_key("likedBy"):
                usrs = post['likedBy']    
            else: usrs = []
            url = foundUrl.groups(0)[0]
            urlText = get_text_from_url(url)
            text = post['message']
            content = {
                "url":url,
                "urlText":urlText,
                "text":text,
                "likes":usrs,
                "date":post['created'],
                "id":post['id'] + ": "+name 
                }
            urls.append(content)
    if len(post['comments'])>0: 
        urls += list(chain(*map(getContentFlat,post['comments'])))
    urls = filter(lambda x: len(x)>0,urls)
    if len(urls)>1:
        urls[0]['text'] = " ".join(u['text'] for u in urls)
        urls[0]['urlText'] = " ".join(u['urlText'] for u in urls)
        urls[0]['likes'].extend(chain(*[u['likes'] for u in urls if u['likes']]))
        urls = [urls[0]]
    return urls

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
                    

class fbTraverserContentFlat(object):
    def __init__(self, fn):
        self.fn = fn

    def __iter__(self):
        with gzip.open(self.fn) as fh:
            for line in fh:
                yield getContent(json.loads(line, strict=False))
                    



