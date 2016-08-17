import urllib
import urllib.request
import json

BASEURL = "https://manifesto-project.wzb.eu/tools/"
VERSION = "MPDS2016a"
APIKEY  = AN API KEY STRING FROM https://manifestoproject.wzb.eu/information/documents/api 
COUNTRY = "Germany"

def get_url(url):
    return urllib.request.urlopen(url).read().decode()

def api_get_latest_text_version(text_id):
    '''
    Get the latest version id of this text
    '''
    versionsUrl = BASEURL+"api_list_metadata_versions.json?keys[]="+text_id+"&api_key="+APIKEY
    versions = json.loads(get_url(versionsUrl))
    return versions['versions'][-1]

def api_get_manifesto_id(text_id,version):
    '''
    Get manifesto id of a text given the text id and a version id
    '''
    textKeyUrl = BASEURL+"api_metadata.json?keys[]="+text_id+"&version="+version+"&api_key="+APIKEY
    textMetaData = json.loads(get_url(textKeyUrl))
    return textMetaData['items'][0]['manifesto_id']

def api_get_core(version = VERSION):
    '''
    Downloads core data set, including information about all parties
    https://manifestoproject.wzb.eu/information/documents/api
    '''
    url = BASEURL + "/api_get_core?key=" + VERSION + "&api_key=" + APIKEY
    return json.loads(get_url(url))

def api_get_text_keys(country=COUNTRY):
    d = api_get_core()
    return [p[5:7] for p in d if p[1]==country]

def api_get_text(text_id):
    '''
    Retrieves the latest version of the manifesto text with corresponding labels
    '''
    # get the latest version of this text
    version = api_get_latest_text_version(text_id)
    # get the text metadata and manifesto ID
    manifestoId = api_get_manifesto_id(text_id,version) 
    textUrl = BASEURL + "api_texts_and_annotations.json?keys[]="+manifestoId+"&version="+version+"&api_key="+APIKEY
    textData = json.loads(get_url(textUrl))
    try:
        return [(t['cmp_code'],t['content']) for t in textData['items'][0]['items']]
    except: 
        print('Could not get text %s'%text_id)
    
def api_get_texts(country=COUNTRY):
    # get all tuples of party/date corresponding to a manifesto text in this country
    textKeys = api_get_text_keys(country)
    # get the texts
    texts = {t[1]+"_"+t[0]:api_get_text(t[1]+"_"+t[0]) for t in textKeys}
    texts = {k: v for k, v in texts.items() if v}
    print("Downloaded %d/%d annotated texts"%(len(texts),len(textKeys)))
    return texts

