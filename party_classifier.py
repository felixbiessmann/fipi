# -*- coding: utf-8 -*-
import pickle
from scipy import ones,hstack,arange,reshape,zeros,random
import json
import os
import glob
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn import metrics

MINLEN = 500

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

def nullPrediction(parties=['linke','gruene','spd','cducsu']):
    return dict([(k, 1.0/len(parties)) for k in parties])

def get_raw_text_bundestag(folder="data/out", legislationPeriod=17):
    '''
    Loads raw text and labels from bundestag sessions 
    (Downloaded preprocessed with https://github.com/bundestag/plpr-scraper, saved in json)
    '''
    data = []
    labels = []
    for f in glob.glob(folder+'/%d*.json'%legislationPeriod):
        speeches = json.load(open(f))
        for speech in speeches:
           if speech['type']=='speech' and \
            speech['speaker_party'] is not None and \
            len(speech['text']) > MINLEN:
                data.append(speech['text'])
                labels.append(speech['speaker_party'])
    return data,labels 

def get_raw_text(folder="data", parties = ['gruene','cducsu','spd','linke']):
    '''
    Loads raw text and labels from manifestoproject csv files 
    (Downloaded from https://visuals.manifesto-project.wzb.eu)
    '''
    partyIds = [str(partyManifestoMap[p]) for p in parties]
    files = glob.glob(folder+"/[0-9]*_2009.csv")
    files = filter(lambda x: x.split('/')[-1].split('_')[0] in partyIds,files)
    return zip(*chain(*filter(None,map(csv2DataTuple,files))))

def csv2DataTuple(f):
    '''
    Extracts list of tuples of (text,label) for each manifestoproject file
    '''
    df = pd.read_csv(f)
    df['content'] = df['content'].astype('str')
    mask = (df['content'].str.len() > MINLEN)
    df = df.loc[mask]
    partyId = f.split('/')[-1].split('_')[0]
    party = [k for (k,v) in partyManifestoMap.items() if str(v) == partyId]
    return zip(df['content'].tolist(), party * len(df))

class PartyClassifier:

    def __init__(self,\
        text, labels ,\
        # the scikit learn pipeline for vectorizing, normalizing and classifying text 
        text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf',LogisticRegression(class_weight='auto',dual=True))]),\
        parameters = {'vect__ngram_range': [(1, 1)],\
               'tfidf__use_idf': (True,False),\
               'clf__C': (10.**arange(-3,6,1.)).tolist()}  
         ):
        
        '''
        Creates a PartyClassifier object
        if no model is found, or train is set True, a new classifier is learned

        INPUT
        folder  the root folder with the raw text data, where the model is stored
        train   set True if you want to train 

        '''
        # if there is no classifier file or training is invoked
        if (not os.path.isfile('party_classifier.pickle')) or train:
            print('Training party classifier')
            self.train(text, data, text_clf=text_clf, parameters=parameters)
        print('Loading party classifier')
        self.clf = pickle.load(open('party_classifier.pickle'))['clf']

    def predict(self,text):
        '''
        Uses scikit-learn Bag-of-Word extractor and classifier and
        applies it to some text. 

        INPUT
        text    a string to assign to a manifestoproject label
        
        '''
        if (not type(text) is list) & (len(text)<3): 
            return nullPrediction()
        # make it a list, if it is a string
        if not type(text) is list: text = [text]
        # remove digits
        text = map(lambda y: filter(lambda x: not x.isdigit(),y),text)
        # predict probabilities
        probabilities = self.clf.predict_proba(text).flatten()
        predictions = dict(zip(self.clf.steps[-1][1].classes_, probabilities.tolist()))
        
        # transform the predictions into json output
        return predictions
   
    def train(self,data, labels, folds = 2, \
        # the scikit learn pipeline for vectorizing, normalizing and classifying text 
        text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf',LogisticRegression(class_weight='auto'))]),\
        parameters = {'vect__ngram_range': [(1, 1)],\
               'tfidf__use_idf': (True,False),\
               'clf__C': (10.**arange(-2,3,1.)).tolist()}  
         ):
        '''
        trains a classifier on the bag of word vectors

        INPUT
        folds   number of cross-validation folds for model selection 

        '''
        ridx = random.permutation(len(labels)) 
        data = [data[i] for i in ridx]
        labels = [labels[i] for i in ridx]
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, cv=StratifiedKFold(labels, folds), n_jobs=-1,verbose=4)
        gs_clf.fit(data,labels)
        # dump classifier to pickle
        pickle.dump({'clf':gs_clf.best_estimator_,'score':gs_clf.best_score_},\
            open('party_classifier.pickle','wb'),-1)

