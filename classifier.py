# -*- coding: utf-8 -*-
from scipy import ones,hstack,arange,reshape,zeros,setdiff1d
import pickle
import json
import os
import glob
from itertools import chain
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

# manifestoproject codes for left/right orientation
label2rightleft = {
    'right': [104,201,203,305,401,402,407,414,505,601,603,605,606],
    'left': [103,105,106,107,403,404,406,412,413,504,506,701,202]
    }

# manifestoproject codes (integer divided by 100) for political domain
label2domain = {
    'External Relations':1,
    'Freedom and Democracy':2,
    'Political System':3,
    'Economy':4,
    'Welfare and Quality of Life':5,
    'Fabric of Society':6
    }

def saveLabelSchema(folder = ""):
    domainlabels = [{"domain":x[0],"label":x[1]} for x in label2domain.items()]
    manifestocodes = [{"manifestocode":x[0],"topic":x[1]} for x in manifestolabels(folder).items()]
    
    json.dump(
        {
            "leftright":label2rightleft,
            "domain":domainlabels,
            "manifestocodes":manifestocodes 
        },\
        open(folder+"/schema.json","wb"),\
        sort_keys=True, indent=2,separators=(',', ': '))

def nullPrediction(folder = "data/manifesto"):
    return json.load(open(folder+"/nullPrediction.json"))

def manifestolabels(folder = "data/manifesto"):
    lines = open(folder+"/manifestolabels.txt").readlines()
    return dict(map(lambda x: (int(x[3:6]), x[8:-2]),lines))

mc = manifestolabels()

def get_raw_text(folder="data/manifesto"):
    '''
    Loads raw text and labels from manifestoproject csv files 
    (Downloaded from https://visuals.manifesto-project.wzb.eu)
    '''
    files = glob.glob(folder+"/[0-9]*_[0-9]*.csv")
    return zip(*chain(*filter(None,map(csv2DataTuple,files))))

def csv2DataTuple(f):
    '''
    Extracts list of tuples of (text,label) for each manifestoproject file
    '''
    df = pd.read_csv(f,quotechar="\"").dropna()
    return zip(df['content'].tolist(),df['cmp_code'].map(int).tolist())

def mapLabel2RightLeft(label):
    '''
    Maps single manifestoproject label (buest guess of classifier)
    to left right label (non-probabilistic)
    '''
    return dict(map(lambda x: (x[0],label in x[1]),label2rightleft.items()))

def mapLabel2Domain(label):
    '''
    Maps single manifestoproject label (buest guess of classifier)
    to domain label (non-probabilistic)
    '''
    return dict(map(lambda x: (x[0],label/100 is x[1]),label2domain.items()))

def mapPredictions2RightLeft(pred):
    '''
    Maps multivariate probablistic manifestoproject label prediction
    to left right label (probabilistic)
    '''
    rightLeftPred = {label[0]:\
    sum(map(lambda y: y[1],filter(lambda x: x[0] in label[1],pred.items()))) \
            for label in label2rightleft.items()}
    normalizer = sum(rightLeftPred.values())
    if normalizer == 0:
        return [{"label":x[0],"prediction":1.0/len(rightLeftPred)} \
            for x in rightLeftPred.items()]
    else:
        return [{"label":x[0],"prediction":x[1]/normalizer} \
            for x in rightLeftPred.items()]


def mapPredictions2Domain(pred):
    '''
    Maps multivariate probablistic manifestoproject label prediction
    to domain label (probabilistic)
    '''
    domainPred = {label[0]:\
    sum(map(lambda y: y[1],filter(lambda x: x[0]/100 == label[1],pred.items()))) \
            for label in label2domain.items()}
    normalizer = sum(domainPred.values())
    if normalizer == 0:
        return [{"label":x[0],"prediction":1.0/len(domainPred)} \
            for x in domainPred.items()]
    else:
        return [{"label":x[0],"prediction":x[1]/normalizer} \
            for x in domainPred.items()]

class Classifier:

    def __init__(self,train=False):
        '''
        Creates a classifier object
        if no model is found, or train is set True, a new classifier is learned

        INPUT
        folder  the root folder with the raw text data, where the model is stored
        train   set True if you want to train 

        '''
        # if there is no classifier file or training is invoked
        if (not os.path.isfile('classifier.pickle')) or train:
            print('Training classifier')
            self.train()
        print('Loading classifier')
        self.clf = pickle.load(open('classifier.pickle','rb'))

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
        # predict probabilities
        probabilities = self.clf.predict_proba(text).flatten()
        predictions = dict(zip(self.clf.classes_, probabilities.tolist()))
        
        # transform the predictions into json output
        return {
                'text':text,
                'leftright':mapPredictions2RightLeft(predictions),
                'domain':mapPredictions2Domain(predictions),
                'manifestocode':[{"label":mc[x[0]],"prediction":x[1]} for x in predictions.items()]
                }
   
    def train(self,folds = 2):
        '''
        trains a classifier on the bag of word vectors

        INPUT
        folds   number of cross-validation folds for model selection 

        '''
        try:
            # load the data
            data,labels = get_raw_text()
        except:
            print('Could not load text data file in\n')
            raise
        # the scikit learn pipeline for vectorizing, normalizing and classifying text 
        text_clf = Pipeline([('vect', CountVectorizer()),
                            ('tfidf', TfidfTransformer()),
                            ('clf',LogisticRegression(class_weight='balanced',dual=True))])
        parameters = {'vect__ngram_range': [(1, 1)],\
               'tfidf__use_idf': (True,False),\
               'clf__C': (10.**arange(4,5,1.)).tolist()}  
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, cv=folds, n_jobs=-1,verbose=3)
        gs_clf.fit(data,labels)
        # dump classifier to pickle
        pickle.dump(gs_clf.best_estimator_,open('classifier.pickle','wb'))

