#from plpr_parser.scraper import *
import glob
import os
import json
import pdb
import scipy as sp
import classifier

DATA_PATH = os.environ.get('DATA_PATH', 'data')
TXT_DIR = os.path.join(DATA_PATH, 'txt')
OUT_DIR = os.path.join(DATA_PATH, 'out')
clf = classifier.Classifier(train=False)

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

def classify_speeches():
    
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
        print "processing %d speeches in %s"%(len(speeches),f)
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
        print "processing %d speeches in %s"%(len(speeches),f)
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
    print "Loading transcript: %s/%.3d, from %s" % (wp, session, filename)
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

 
