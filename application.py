# -*- coding: utf8 -*-
import json
from flask import Flask, render_template
from random import choice

# EB looks for an 'application' callable by default.
app = Flask(__name__)


@app.route("/")
def template_test():
    with open("../CDU-enriched.json.gzip", "r") as f:
        d = f.read()
    posts = "[" + d.replace("}{", "},{") + "]"
    posts = json.loads(posts)
    # p = posts[0]
    p = choice(posts)
    text = p['text'][0]

    sort_labels = sorted(p['manifestocode'], key=lambda x: x.get('prediction'))
    labels = [l['label'][:-2] for l in sort_labels]
    list_1 = labels[:6]
    list_2 = labels[6:12]
    context = dict(my_string=text, list_1=list_1,list_2=list_2)
    return render_template('index.html', **context)



# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
