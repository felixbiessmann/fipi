# -*- coding: utf8 -*-
import os
import json
import sqlite3
from flask import Flask, render_template, g, request #, session,  redirect, url_for, abort, flash

from sqlalchemy_wrapper import SQLAlchemy
from random import choice, shuffle, randint
from models import PartyPost, PostLabel, HumanPostLabel, db, categories, DB_FILE

# EB looks for an 'application' callable by default.
app = Flask(__name__)
app.config.from_object(__name__)

db.init_app(app)

# Load default config and override config from an environment variable
app.config['DATABASE'] = os.path.join(app.root_path, DB_FILE)

def load_human_label(form_data):
    post = form_data.get('post_id')
    human = form_data.get('human_id')
    for k in form_data.keys():
        if 'category' in k:
            labels = form_data.getlist(k)
            category = k.split('_')[1]
            for label in labels:
                human_label = HumanPostLabel(post_id=post, human_id=human, category=category, label=label)
                try:
                    db.add(human_label)
                except:
                    db.session.rollback()
                finally:
                    db.session.commit()

@app.route("/", methods=('GET', 'POST'))
def template_test():

    if request.method == 'POST':
        load_human_label(request.form)

    posts = db.query(PartyPost).all()
    p = choice(posts)
    text = p.post_text

    domain_labels = [(l.label, l.prediction) for l in p.labels if l.category == 'domain']
    sort_domain_labels = sorted(domain_labels, key=lambda x: x[1], reverse=True)
    manifestocode_labels = [(l.label, l.prediction) for l in p.labels if l.category == 'manifestocode']
    sort_manifestocode_labels = sorted(manifestocode_labels, key=lambda x: x[1], reverse=True)
    list_1 = [i[0] for i in sort_domain_labels[:6]]
    list_2 = [i[0] for i in sort_manifestocode_labels[:6]]
    shuffle(list_1)
    shuffle(list_2)

    human = randint(100,200) # place holder for human tracking;
    context = dict(post_text=text, list_1=list_1,list_2=list_2, post_id=p.post_id, human_id=human)
    return render_template('index.html', **context)

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run()
