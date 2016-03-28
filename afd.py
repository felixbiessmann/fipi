from party_classifier import PartyClassifier
import codecs
from classifier import Classifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, additive_chi2_kernel
from itertools import chain
import scipy
import pdb
import json
import re

parties = ['gruene', 'pirates', 'linke', 'fdp', 'cducsu', 'spd']

pclf = PartyClassifier(train=False)#True,parties=parties)
mclf = Classifier(train=False)#True)

DATA = '/Users/felix/Code/Python/fipi/data/parteiprogramme-niels/'
MINLEN = 200 # minimum length of units
manifestos = {
    'Afd':'afd-parteiprogramm.txt',
    'CDUCSU':'cdu-parteiprogramm.txt',
    'Gruene':'gruene-parteiprogramm.txt',
    'SPD':'spd-parteiprogramm.txt',
    'NPD':'npd-parteiprogramm.txt',
    'Linke':'linke-parteiprogramm.txt',
    'FDP':'fdp-parteiprogramm.txt'
    }

manifestoTexts = {k:filter(lambda x: len(x)>MINLEN, open(DATA + v).read().split("\n\n")) for k,v in manifestos.items()}
#manifestoTexts = {k:filter(lambda x: len(x)>MINLEN, re.split('[\.\!\?]',open(DATA + v).read())) for k,v in manifestos.items()}

stops = map(lambda x:x.lower().strip(),codecs.open('data/stopwords.txt').readlines()[6:])
#bowModel = CountVectorizer(max_df=0.1, min_df=2).fit(chain(*manifestoTexts.values()))
bowModel = TfidfVectorizer(min_df=1,max_df=0.01, stop_words=stops).fit(chain(*manifestoTexts.values()))
manifestoBow = {k:bowModel.transform(v) for k,v in manifestoTexts.items()}

svd = TruncatedSVD(n_components=100, random_state=42).fit(bowModel.transform(chain(*manifestoTexts.values())))

matchTexts = [{'name':k,'class':k.lower(),'units':[{'t':vi} for vi in v]} for k,v in manifestoTexts.items() if k is not 'Afd']

idx2word = dict(zip(bowModel.vocabulary_.values(),bowModel.vocabulary_.keys()))


def prediction2vec(p):
    return scipy.array([x['prediction'] for x in sorted(p['manifestocode'], key=lambda k: k['label'])])

manifestoCodePredictions = {k:scipy.vstack([prediction2vec(mclf.predict(t)) for t in v]) for k,v in manifestoTexts.items()}


def process_afd():
    predictions = []
    matches = []
    for afdStatementIdx,statement in enumerate(manifestoTexts['Afd']):
        partyPrediction = pclf.predict(statement)
        manifestoPrediction = mclf.predict(statement)
        item = {
            't': statement, 
            'fipi': manifestoPrediction, 
            'party': partyPrediction}
        predictions.append(item)
        match = [0, 0, 0.0] # [partyidx, textunitidx, score]
        for otherPartyIdx,party in enumerate(matchTexts):
            #bowSim = cosine_similarity(svd.transform(manifestoBow[party['name']]), svd.transform(manifestoBow['Afd'][afdStatementIdx,:])).flatten()
            bowSim = cosine_similarity(scipy.hstack([manifestoCodePredictions[party['name']],svd.transform(manifestoBow[party['name']])]), scipy.hstack([manifestoCodePredictions['Afd'][afdStatementIdx,:],svd.transform(manifestoBow['Afd'][afdStatementIdx,:]).flatten()])).flatten()
            maxInd = bowSim.argmax()
            if bowSim[maxInd] > match[2]:
                match = [otherPartyIdx, maxInd, bowSim[maxInd]]
        #statementBow = bowModel.transform([statement])
        #print('*******************************')
        #matchBow = bowModel.transform([matchTexts[match[0]]['units'][match[1]]['t']])
        #print([idx2word[x] for x in set(matchBow.nonzero()[1]).intersection(set(statementBow.nonzero()[1]))])
        #print(statement)
        #print('**************')
        #print(matchTexts[match[0]]['units'][match[1]]['t'])
        #print('**************')
        #pdb.set_trace()

        matches.append([[afdStatementIdx], [match[0], [match[1]]], match[2]])
            
    totalDump = {
    'main_text': 
        {
          'name':'Parteiprogramm AfD',
          'class':'afd',
          'units':predictions
          },
     'match_texts': matchTexts,
     'matchings':matches
    }
    json.dump(totalDump,open(DATA+'totalDump.json','wb'))
    return totalDump
