import facebook, os
import pandas as pd

PARTIES = {
        'AfD':'alternativefuerde',
        'CDU':'CDU',
        'SPD':'SPD',
        'Gruene':'B90DieGruenen',
        'Linke':'linkspartei',
        'NPD':'npd.de',
        'FDP':'FDP'
        }

def get_credentials():
    app_id = os.environ["FACEBOOK_APP_ID"]
    app_secret = os.environ["FACEBOOK_SECRET"]
    return app_id,app_secret

def get_posts(fbPage,app_id,app_secret):
    fbapi = facebook.GraphAPI()
    token = fbapi.get_app_access_token(app_id, app_secret)
    graph = facebook.GraphAPI(access_token=token)
    party = graph.get_object(fbPage)
    return graph.get_connections(party['id'], 'posts')

def get_fb_texts(download=True, partyPages=PARTIES, folder="data/fb"):
    app_id, app_secret = get_credentials()
    alldfs = []
    for party,fbPartyPage in partyPages.items():
        fn = folder + "/fb-%s.csv"%party
        if os.path.isfile(fn) and (download==False):
            print("Loading %s"%fn)
            df = pd.read_csv(fn)
        else:
            print("Downloading texts from fb for %s."%party)
            texts = get_posts(fbPartyPage, app_id, app_secret)
            df = pd.DataFrame(texts['data'])
            df['party'] = party
            df.to_csv(fn,index=False)
        alldfs.append(df)

    return pd.concat(alldfs, axis=0, ignore_index=True)
