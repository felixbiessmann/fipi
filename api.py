# -*- coding: utf-8 -*-
import flask
from flask import Flask, request, jsonify, render_template
import os
import urllib
from classifier import Classifier
from retrying import retry
from bs4 import BeautifulSoup
from newsreader import get_news
from apscheduler.schedulers.background import BackgroundScheduler
import json

app = Flask(__name__)

DEBUG = os.environ.get('DEBUG') != None
VERSION = 0.1

# Schedules news reader to be run at 00:00
#scheduler = BackgroundScheduler()
#scheduler.add_job(get_news, 'interval', minutes=360)
#scheduler.start()

@retry(stop_max_attempt_number=5)
def fetch_url(url):
    '''
    get url with readability
    '''
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html,"lxml")
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if len(chunk.split(" "))>1)
    return text.encode('utf-8')

### API
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
        text = fetch_url(url)
        return jsonify(classifier.predict(text))
    else:
        text = request.form['text']
        return jsonify(classifier.predict(text))

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
    classifier = Classifier(train=True)
    get_news()
    # Open a web browser pointing at the app.
    os.system("open http://localhost:{0}/".format(port))
    app.run(host='0.0.0.0', port = port, debug = DEBUG)
