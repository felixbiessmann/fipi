# -*- coding: utf-8 -*-
from sklearn.decomposition import KernelPCA
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats.mstats import zscore
import glob
import json
import re
import datetime
import os
import cPickle
import codecs
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import double,zeros

def get_news(sources=['spiegel','faz','welt','zeit']):
    '''
    Collects all news articles from political ressort of major German newspapers
    Articles are transformed to BoW vectors and assigned to a political party
    For better visualization, articles' BoW vectors are also clustered into topics

    INPUT
    folder      the model folder containing classifier and BoW transformer
    sources     a list of strings for each newspaper for which a crawl is implemented
                default ['zeit','sz']

    '''
    import classifier
    from bs4 import BeautifulSoup
    from api import fetch_url
    import urllib2
    
    articles = []
    
    # the classifier for prediction of political attributes 
    clf = classifier.Classifier(train=False)
    
    for source in sources:

        if source is 'spiegel':
            # fetching articles from sueddeutsche.de/politik
            url = 'http://www.spiegel.de/politik'
            site = BeautifulSoup(urllib2.urlopen(url).read())
            titles = site.findAll("div", { "class" : "teaser" })
            urls = ['http://www.spiegel.de'+a.findNext('a')['href'] for a in titles]
         
        if source is 'faz':
            # fetching articles from sueddeutsche.de/politik
            url = 'http://www.faz.net/aktuell/politik'
            site = BeautifulSoup(urllib2.urlopen(url).read())
            titles = site.findAll("a", { "class" : "TeaserHeadLink" })
            urls = ['http://www.faz.net'+a['href'] for a in titles]
         
        if source is 'welt':
            # fetching articles from sueddeutsche.de/politik
            url = 'http://www.welt.de/politik'
            site = BeautifulSoup(urllib2.urlopen(url).read())
            titles = site.findAll("a", { "class" : "as_teaser-kicker" })
            urls = [a['href'] for a in titles]
         
        if source is 'sz-without-readability':
            # fetching articles from sueddeutsche.de/politik
            url = 'http://www.sueddeutsche.de/politik'
            site = BeautifulSoup(urllib2.urlopen(url).read())
            titles = site.findAll("div", { "class" : "teaser" })
            urls = [a.findNext('a')['href'] for a in titles]
       
        if source is 'zeit':
            # fetching articles from zeit.de/politik
            url = 'http://www.zeit.de/politik'
            site = BeautifulSoup(urllib2.urlopen(url).read())
            urls = [a['href'] for a in site.findAll("a", { "class" : "teaser-small__combined-link" })]

        print "Found %d articles on %s"%(len(urls),url)
         
        # predict party from url for this source
        print "Predicting %s"%source
        for url in urls:
            try:
                title,text = fetch_url(url)
                prediction = clf.predict(text)
                prediction['url'] = url
                prediction['source'] = source
                articles.append((title,prediction))
            except:
                print('Could not get text from %s'%url)
                pass

    # do some topic modeling
    topics = kpca_cluster(map(lambda x: x[1]['text'][0], articles))
    
    # store current news and topics
    json.dump(articles,open('news.json','wb'))
    json.dump(topics,open('topics.json','wb'))

def load_sentiment(negative='SentiWS_v1.8c/SentiWS_v1.8c_Negative.txt',\
        positive='SentiWS_v1.8c/SentiWS_v1.8c_Positive.txt'):
    words = dict()
    for line in open(negative).readlines():
        parts = line.strip('\n').split('\t')
        words[parts[0].split('|')[0]] = double(parts[1])
        if len(parts)>2:
            for inflection in parts[2].strip('\n').split(','):
                words[inflection] = double(parts[1])
    
    for line in open(positive).readlines():
        parts = line.strip('\n').split('\t')
        words[parts[0].split('|')[0]] = double(parts[1])
        if len(parts)>2:
            for inflection in parts[2].strip('\n').split(','):
                words[inflection] = double(parts[1])
   
    return words

def get_sentiments(data):
    
    # filtering out some noise words
    stops = map(lambda x:x.lower().strip(),open('stopwords.txt').readlines()[6:])

    # vectorize non-stopwords 
    bow = TfidfVectorizer(min_df=2,stop_words=stops)
    X = bow.fit_transform(data)

    # map sentiment vector to bow space
    words = load_sentiment()
    sentiment_vec = zeros(X.shape[1])
    for key in words.keys():
        if bow.vocabulary_.has_key(key):
            sentiment_vec[bow.vocabulary_[key]] = words[key]
    
    # compute sentiments 
    return X.dot(sentiment_vec)


def kpca_cluster(data,nclusters=20,topwhat=10):
    '''

    Computes clustering of bag-of-words vectors of articles

    INPUT
    folder      model folder
    nclusters   number of clusters

    '''
    from sklearn.cluster import KMeans
    # filtering out some noise words
    stops = map(lambda x:x.lower().strip(),codecs.open('data/stopwords.txt',"r","utf-8").readlines()[6:])

    # vectorize non-stopwords 
    bow = TfidfVectorizer(min_df=4,stop_words=stops)
    X = bow.fit_transform(data)

    # creating bow-index-to-word map
    idx2word = dict(zip(bow.vocabulary_.values(),bow.vocabulary_.keys()))
   
    # compute clusters
    km = KMeans(n_clusters=nclusters).fit(X)

    clusters = []
    for icluster in range(nclusters):
        nmembers = (km.labels_==icluster).sum()
        if nmembers > 1: # only group clusters big enough but not too big
            members = (km.labels_==icluster).nonzero()[0]
            topwordidx = km.cluster_centers_[icluster,:].argsort()[-topwhat:][::-1]
            topwords = ' '.join([idx2word[wi] for wi in topwordidx])
            #print u'Cluster %d'%icluster + u' %d members'%nmembers + u'\n\t'+topwords
            clusters.append({
                'name':'Cluster-%d'%icluster,
                'description': topwords,
                'members': list(members),
                })

    return clusters

def write_distances_json(folder='model'):
    articles, data = get_news()
    distances_json = {
            'articles': articles,
            'distances': [
                { 'name': dist, 'distances': pairwise_dists(data) } for dist in dists
            ],
            'clusterings': [
                { 'name': 'Parteivorhersage', 'clusters': party_cluster(articles) },
                { 'name': 'Ähnlichkeit', 'clusters': kpca_cluster(data,nclusters=len(articles)/2,ncomponents=40,zscored=False) },
            ]
        }

    # save article with party prediction and distances to closest articles
    datestr = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    open(folder+'/distances-%s'%(datestr)+'.json', 'wb').write(json.dumps(distances_json))
    # also save that latest version for the visualization
    open(folder+'/distances.json', 'wb').write(json.dumps(distances_json))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(\
        description='Downloads, transforms and clusters news articles')

    parser.add_argument('-p','--distances',help='If pairwise distances of text should be computed',\
            action='store_true', default=True)
    
    args = vars(parser.parse_args())
    if args['distances']:
        write_distances_json(folder=args['folder'])
