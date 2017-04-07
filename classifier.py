# -*- coding: utf-8 -*-
from scipy import ones,hstack,arange,reshape,zeros,setdiff1d,floor
import pickle
import json
import os
import glob
from itertools import chain
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics
from manifesto_data import get_manifesto_texts
from operator import itemgetter
import numpy as np

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

# pd.concat([df,clf.predictBatch(df.message.fillna(''))])

def manifestolabels(folder = "data/manifesto"):
    lines = open(folder+"/manifestolabels.txt").readlines()
    return dict(map(lambda x: (int(x[3:6]), x[8:-2]),lines))

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
        text = ["".join([x for x in t if not x.isdigit()]) for t in text]
        probabilities = self.clf.predict_proba(text).flatten()
        predictionsManifestocode = dict(zip(self.clf.classes_, probabilities.tolist()))
        predictionsDomain = {l:sum(probabilities[np.floor(self.clf.classes_/100) == idx]) for l,idx in label2domain.items()}
        predictionsRight = sum([p for l,p in predictionsManifestocode.items() if l in label2rightleft['right']])
        predictionsLeft = sum([p for l,p in predictionsManifestocode.items() if l in label2rightleft['left']])
        # transform the predictions into json output
        return {
                'leftright':{'right':predictionsRight,'left':predictionsLeft},
                'domain':predictionsDomain,
                'manifestocode':{mc[x[0]]:x[1] for x in predictionsManifestocode.items()}
                }
    def predictBatch(self,texts):
        '''
        Uses scikit-learn Bag-of-Word extractor and classifier and
        applies it to some text.

        INPUT
        text    a string to assign to a manifestoproject label

        '''
        mc = manifestolabels()
        df = pd.DataFrame(self.clf.predict_proba(texts),columns=self.clf.classes_)
        mcCols = df.columns
        for dom,domIdx in label2domain.items():
            df[dom] = df[mcCols[floor(mcCols/100)==domIdx]].sum(axis=1)
        df['right'] = df[label2rightleft['right']].sum(axis=1)
        df['left'] = df[label2rightleft['left']].sum(axis=1)
        return df.rename(index=str,columns=mc)


    def train(self,folds = 2):
        '''
        trains a classifier on the bag of word vectors

        INPUT
        folds   number of cross-validation folds for model selection

        '''
        try:
            # load the data
            data,labels = get_manifesto_texts()
        except:
            print('Could not load text data file in\n')
            raise
        # the scikit learn pipeline for vectorizing, normalizing and classifying text
        text_clf = Pipeline([('vect', HashingVectorizer()),
                            ('clf',SGDClassifier(loss="log"))])
        parameters = {'vect__ngram_range': [(1, 2)],
               'clf__alpha': (10.**arange(-5,-4)).tolist()}
        # perform gridsearch to get the best regularizer
        gs_clf = GridSearchCV(text_clf, parameters, cv=folds, n_jobs=-1,verbose=3)
        gs_clf.fit(data,labels)
        # dump classifier to pickle
        pickle.dump(gs_clf.best_estimator_,open('classifier.pickle','wb'))
