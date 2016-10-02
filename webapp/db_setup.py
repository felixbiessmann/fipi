exit()
python
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

db = SQLAlchemy('sqlite:///:test_fb:')


party_post = PartyPost(
                    post_id=p.get('postId'),
                    category=p['domain'][0].get('label'),
                    label=p['domain'][0].get('label'),
                    prediction=p['domain'][0].get('prediction'),
                    number_comments=p.get('numberComments'),
                    number_likes=p.get('numberLikes'),
                    party=p.get('party'),
                    post_text=p.get('text'),
                    timestamp=convert_from_unix_time(p['timeStamp']) if p.get('timestamp') else None
                    )

db.session.merge(party_post)


def convert_from_unix_time(unix_time):
    # if string is desired then | .strftime('%Y-%m-%d %H:%M:%S')
    return datetime.datetime.fromtimestamp(int(unix_time))

def hash_id(*args):
    some_args = "".join(str(args))
    id_hash = hashlib.md5(some_args).hexdigest()
    return id_hash

class PartyPost(db.Model):
    def __init__(self, **kwargs):
        if 'hash_id' not in kwargs:
            kwargs['hash_id'] = hash_id(kwargs['post_id'], kwargs['category'], kwargs['label'])
            self.id = kwargs['hash_id']
        super(PartyPost, self).__init__(**kwargs)
    hash_id = db.Column(db.String(63), primary_key=True)
    post_id = db.Column(db.String(63))
    category = db.Column(db.String(63))
    label = db.Column(db.String(123))
    prediction = db.Column(db.Float)
    number_comments = db.Column(db.Integer)
    number_likes = db.Column(db.Integer)
    party = db.Column(db.String)
    post_text = db.Column(db.Text)
    timestamp = db.Column(db.DateTime(timezone=False))


def load_post_file(filename):
    with open(filename, "r") as f:
        d = f.read()
    posts = "[" + d.replace("}{", "},{") + "]"
    posts = json.loads(posts)
    for p in posts:




"""domain --> list(len=6) --> label --> <type 'unicode'> --> final_value",
 "domain --> list(len=6) --> prediction --> <type 'float'> --> final_value",
 "leftright --> list(len=2) --> label --> <type 'unicode'> --> final_value",
 "leftright --> list(len=2) --> prediction --> <type 'float'> --> final_value",
 "manifestocode --> list(len=56) --> label --> <type 'unicode'> --> final_value",
 "manifestocode --> list(len=56) --> prediction --> <type 'float'> --> final_value",
 "numberComments --> <type 'unicode'> --> final_value",
 "numberLikes --> <type 'unicode'> --> final_value",
 "party --> <type 'unicode'> --> final_value",
 "postId --> <type 'unicode'> --> final_value",
 "text --> list(len=1) --> <type 'unicode'> --> final_value",
 "timeStamp --> <type 'unicode'> --> final_value"]"""

db


db.create_all()
db.drop_all()

todos = db.query(ToDo).all()
