exit()
python

# -*- coding: utf8 -*-

import json
import gzip
import hashlib
import re
import datetime
import time
from sqlalchemy_wrapper import SQLAlchemy

with open("CDU-enriched.json.gzip", "r") as f:
    d = f.read()

posts = "[" + d.replace("}{", "},{") + "]"
posts = json.loads(posts)
p = posts[0]

def convert_from_unix_time(unix_time):
    # if string is desired then | .strftime('%Y-%m-%d %H:%M:%S')
    return datetime.datetime.fromtimestamp(int(unix_time))

def hash_id(*args):
    some_args = "".join(str(args))
    id_hash = hashlib.md5(some_args).hexdigest()
    return id_hash

db = SQLAlchemy('sqlite:///:test_fb:')

class PartyPost(db.Model):
    def __init__(self, **kwargs):
        # if 'hash_id' not in kwargs:
        #     kwargs['hash_id'] = hash_id(kwargs['post_id'], kwargs['category'], kwargs['label'])
        #     self.id = kwargs['hash_id']
        super(PartyPost, self).__init__(**kwargs)
    # hash_id = db.Column(db.String(63), primary_key=True)
    post_id = db.Column(db.String(63), primary_key=True)
    number_comments = db.Column(db.Integer)
    number_likes = db.Column(db.Integer)
    party = db.Column(db.String)
    post_text = db.Column(db.Text)
    timestamp = db.Column(db.DateTime(timezone=False))
    labels = db.relationship('PostLabel', backref='party_posts', lazy='dynamic')


class PostLabel(db.Model):
    def __init__(self, **kwargs):
        if 'hash_id' not in kwargs:
            kwargs['hash_id'] = hash_id(kwargs['post_id'], kwargs['category'], kwargs['label'])
            self.id = kwargs['hash_id']
        super(PostLabels, self).__init__(**kwargs)
    hash_id = db.Column(db.String(63), primary_key=True)
    post_id = db.Column(db.String(63), db.ForeignKey('PartyPost.post_id'))
    category = db.Column(db.String(63))
    label = db.Column(db.String(123))
    prediction = db.Column(db.Float)


party_post = PartyPost(
                    post_id=p.get('postId'),
                    category='domain',
                    number_comments=p.get('numberComments'),
                    number_likes=p.get('numberLikes'),
                    party=p.get('party'),
                    post_text=p.get('text')[0],
                    timestamp=convert_from_unix_time(p['timeStamp']) if p.get('timestamp') else None
                    )

party_post = PostLabel(
                    post_id=p.get('postId'),
                    category='domain',
                    label=p['domain'][0].get('label'),
                    prediction=p['domain'][0].get('prediction'),
                    number_comments=p.get('numberComments'),
                    number_likes=p.get('numberLikes'),
                    party=p.get('party'),
                    post_text=p.get('text')[0],
                    timestamp=convert_from_unix_time(p['timeStamp']) if p.get('timestamp') else None
                    )


loaded_p = db.session.merge(party_post)

db.add(party_post)
db.session.commit()

db.session.rollback()
db.session.commit()

def load_post_file(filename):
    with open(filename, "r") as f:
        d = f.read()
    posts = "[" + d.replace("}{", "},{") + "]"
    posts = json.loads(posts)
    for p in posts:



db.create_all()
db.drop_all()

all_posts = db.query(PartyPost).all()

p1 = all_posts[1]
p1.post_id
p1.post_text
