import facebook

def getCredentials():
    app_id = os.environ["FACEBOOK_APP_ID"]
    app_secret = os.environ["FACEBOOK_SECRET"]
    return app_id,app_secret

def get_posts(party="CDU",app_id,app_secret):
    fbapi = facebook.GraphAPI()
    token = fbapi.get_app_access_token(app_id, app_secret)
    graph = facebook.GraphAPI(access_token=token)
    party = graph.get_object(partyId)
    return graph.get_object(party['id'], 'posts')
