import urllib, itertools, json, os
import urllib.request
import pandas as pd

BASEURL = "https://manifesto-project.wzb.eu/tools/"
VERSION = "MPDS2016a"
APIKEY  = #AN API KEY STRING FROM https://manifestoproject.wzb.eu/information/documents/api
COUNTRY = "Germany"

def get_url(url):
    return urllib.request.urlopen(url).read().decode()

def get_latest_version():
    '''
    Get the latest version id of the Corpus
    '''
    versionsUrl = BASEURL+"list_metadata_versions.json?&key="+APIKEY
    versions = json.loads(get_url(versionsUrl))
    return versions['versions'][-1]

def get_manifesto_id(text_id,version):
    '''
    Get manifesto id of a text given the text id and a version id
    '''
    textKeyUrl = BASEURL+"metadata.json?keys[]="+text_id+"&version="+version+"&key="+APIKEY
    textMetaData = json.loads(get_url(textKeyUrl))
    return textMetaData['items'][0]['manifesto_id']

def get_core(version = VERSION):
    '''
    Downloads core data set, including information about all parties
    https://manifestoproject.wzb.eu/information/documents/api
    '''
    url = BASEURL + "/get_core?key=" + VERSION + "&key=" + APIKEY
    return json.loads(get_url(url))

def get_text_keys(country=COUNTRY):
    d = get_core()
    return [p[5:7] for p in d if p[1]==country]

def get_text(text_id):
    '''
    Retrieves the latest version of the manifesto text with corresponding labels
    '''
    # get the latest version of this text
    version = get_latest_version()
    # get the text metadata and manifesto ID
    manifestoId = get_manifesto_id(text_id,version)
    textUrl = BASEURL + "texts_and_annotations.json?keys[]="+manifestoId+"&version="+version+"&key="+APIKEY
    textData = json.loads(get_url(textUrl))
    try:
        return [(t['cmp_code'],t['text']) for t in textData['items'][0]['items']]
    except:
        print('Could not get text %s'%text_id)

def get_texts_per_party(country=COUNTRY):
    # get all tuples of party/date corresponding to a manifesto text in this country
    textKeys = get_text_keys(country)
    # get the texts
    texts = {t[1]+"_"+t[0]:get_text(t[1]+"_"+t[0]) for t in textKeys}
    texts = {k: v for k, v in texts.items() if v}
    print("Downloaded %d/%d annotated texts"%(len(texts),len(textKeys)))
    return texts

def get_texts(country=COUNTRY):
    texts = get_texts_per_party(country)
    return [x for x in list(itertools.chain(*texts.values())) if x[0]!='NA' and x[0]!='0']

def get_manifesto_texts(country = "Germany", folder="data/manifesto"):
    fn = folder + "/manifesto-%s.csv"%country
    if os.path.isfile(fn):
        print("Loading %s"%fn)
        df = pd.read_csv(fn)
    else:
        print("Downloading texts from manifestoproject.")
        manifestotexts = get_texts(country)
        df = pd.DataFrame(manifestotexts,columns=['cmp_code','content'])
        df.to_csv(fn,index=False)
    return df['content'].map(str).tolist(),df['cmp_code'].map(int).tolist()
