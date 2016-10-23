# -*- coding: utf8 -*-
import json
import gzip
import hashlib
import re
import datetime
import time
from sqlalchemy_wrapper import SQLAlchemy

DB_FILE = 'test_fb_posts.db'
db = SQLAlchemy('sqlite:///{}'.format(DB_FILE))

categories = ['domain', 'leftright', 'manifestocode']

def convert_from_unix_time(unix_time):
    # if string is desired then | .strftime('%Y-%m-%d %H:%M:%S')
    return datetime.datetime.fromtimestamp(int(unix_time))

def hash_id(*args):
    some_args = "".join(str(args))
    id_hash = hashlib.md5(some_args).hexdigest()
    return id_hash


class PartyPost(db.Model):
    def __init__(self, **kwargs):
        super(PartyPost, self).__init__(**kwargs)
    post_id = db.Column(db.String(63), primary_key=True)
    number_comments = db.Column(db.Integer)
    number_likes = db.Column(db.Integer)
    party = db.Column(db.String)
    post_text = db.Column(db.Text)
    timestamp = db.Column(db.DateTime(timezone=False))
    labels = db.relationship('PostLabel', backref=db.backref('party_posts')) #, lazy='dynamic'
    human_labels = db.relationship('HumanPostLabel', backref=db.backref('party_posts')) #, lazy='dynamic'


class PostLabel(db.Model):
    def __init__(self, **kwargs):
        if 'hash_id' not in kwargs:
            kwargs['hash_id'] = hash_id(kwargs['post_id'], kwargs['category'], kwargs['label'])
            self.id = kwargs['hash_id']
        super(PostLabel, self).__init__(**kwargs)
    hash_id = db.Column(db.String(63), primary_key=True)
    post_id = db.Column(db.String(63), db.ForeignKey('party_posts.post_id'))
    category = db.Column(db.String(63))
    label = db.Column(db.String(123))
    prediction = db.Column(db.Float)


class HumanPostLabel(db.Model):
    def __init__(self, **kwargs):
        if 'hash_id' not in kwargs:
            kwargs['hash_id'] = hash_id(kwargs['post_id'], kwargs['category'], kwargs['label'], kwargs['human_id'])
            self.id = kwargs['hash_id']
        super(HumanPostLabel, self).__init__(**kwargs)
    hash_id = db.Column(db.String(63), primary_key=True)
    post_id = db.Column(db.String(63), db.ForeignKey('party_posts.post_id'))
    category = db.Column(db.String(63))
    label = db.Column(db.String(123))
    human_id = db.Column(db.String(123))
