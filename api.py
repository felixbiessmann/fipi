# -*- coding: utf-8 -*-
import flask,json,os,urllib
from flask import Flask, request, jsonify, render_template
import os,re
import urllib
from classifier import Classifier
from readability.readability import Document
from retrying import retry
from bs4 import BeautifulSoup
from newsreader import get_news
from apscheduler.schedulers.background import BackgroundScheduler
#from coocSentiment import getTextByPartyAND,getPartyTextsDF
import pandas as pd

app = Flask(__name__)

DEBUG = os.environ.get('DEBUG') != None
VERSION = 0.1

# Schedules news reader to be run at 00:00
scheduler = BackgroundScheduler()
scheduler.add_job(get_news, 'interval', minutes=360)
scheduler.start()

@retry(stop_max_attempt_number=5)
def fetch_url(url):
    '''
    get url with readability
    '''
    html = urllib.request.urlopen(url).read()
    readable_article = Document(html).summary()
    title = Document(html).short_title()
    text = BeautifulSoup(readable_article,"lxml").get_text()
    return title,text

### API
@app.route("/api/postquery", methods=['POST'])
def postquery():
    wordlist = re.sub('[\s,]+'," ",request.form['wordlist']).split(" ")
    words = [re.sub('\W+','', w.lower()) for w in wordlist]
    party = request.form['partyRadio']
    print("querying %s for [%s]"%(party,", ".join(words)))
    textdf = getTextByPartyAND(fbtextdf,words,party)
    print("Found %d texts"%len(textdf))
    return jsonify([{"text":t} for t in textdf['text'].values])

@app.route("/api/newstopics")
def newstopics():
    return open('topics.json').read()

@app.route("/api/news")
def news():
    return open('news.json').read()

@app.route("/api")
def api():
    return jsonify(dict(message='political affiliation prediction api', version=VERSION))

@app.route("/api/predict", methods=['POST'])
def predict():
    if 'url' in request.form:
        url = request.form['url']
        title,text = fetch_url(url)
        prediction = classifier.predict(text)
        del prediction['text']
        return jsonify(prediction)
    else:
        text = request.form['text']
        prediction = classifier.predict(text)
        del prediction['text']
        return jsonify(prediction)

# static files
@app.route('/')
def root():
  return app.send_static_file('index.html')

@app.route('/<path:path>')
def static_proxy(path):
  # send_static_file will guess the correct MIME type
  return app.send_static_file(path)

if __name__ == "__main__":
    port = 5000
    classifier = Classifier()
    get_news()
    #fbtextdf = getPartyTextsDF()
    # Open a web browser pointing at the app.
    os.system("open http://localhost:{0}/".format(port))
    app.run(host='0.0.0.0', port = port, debug = DEBUG)
