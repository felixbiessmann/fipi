# -*- coding: utf8 -*-
import json
import os
import gzip
import glob
from models import PartyPost, PostLabel, DB_FILE, db, categories
from flask_script import Command, Manager, Option

class SetupDatabase(Command):

    def __init__(self, path_to_data='../fb_pol_data/raw_data/*-enriched.json.gz', test_load='2'):
        self.path_to_data = path_to_data
        self.test_load = test_load

    def get_options(self):
        return [
            Option('-p', '--path_to_data', dest='path_to_data', default=self.path_to_data),
            Option('-t', '--test_load', dest='test_load', default=self.test_load),
        ]

    def run(self, path_to_data, test_load):
        backup_sqlite_db()
        db.drop_all()
        db.create_all()
        files_to_load = glob.glob(path_to_data)

        if test_load.lower() == 'false':
            print("\n\tFULL LOAD:\tAll posts for each file. This may take a few minutes...\n")
            test_load = False
        else:
            try:
                test_load = int(test_load)
                print("\n\tTEST LOAD:\t{} posts for each file\n".format(test_load))
            except:
                print("\n\t-t (--test_load): enter 'false' to load all posts, or enter an integer to limit the number of posts to load. Default is '2'\n")
                return
        for f in files_to_load:
            f_name = os.path.split(f)[1]
            print("Loading {}".format(f))
            load_posts(f, test_load=test_load)
        print("")

def backup_sqlite_db():
    if os.path.isfile(DB_FILE):
        db_f, db_ext = os.path.splitext(DB_FILE)
        db_backup =  db_f + '_backup.' + db_ext
        with open(DB_FILE, 'r') as read_f:
            f = read_f.read()
            with open(db_backup, 'w') as write_f:
                write_f.write(f)

def load_posts(filename, test_load=False):
    with gzip.open(filename, "rb") as f:
        d = f.read()
    posts = "[" + d.replace("}{", "},{") + "]"
    posts = json.loads(posts)
    posts = posts[:test_load] if test_load else posts
    for p in posts:
        post_id = p.get('postId')
        party_post = PartyPost(
                            post_id=post_id,
                            number_comments=p.get('numberComments'),
                            number_likes=p.get('numberLikes'),
                            party=p.get('party'),
                            post_text=p.get('text')[0],
                            timestamp=convert_from_unix_time(p['timeStamp']) if p.get('timestamp') else None
                            )
        for category in categories:
            cat = p.get(category,[])
            for i in cat:
                post_label = PostLabel(
                                    post_id=post_id,
                                    category=category,
                                    label=i.get('label'),
                                    prediction=i.get('prediction')
                                    )
                try:
                    db.session.merge(post_label)
                    # db.add(post_label)
                except:
                    db.session.rollback()
                finally:
                    db.session.commit()
        try:
            db.session.merge(party_post)
            # db.add(party_post)
        except:
            db.session.rollback()
        finally:
            db.session.commit()


if __name__ == '__main__':
    setup = SetupDatabase()
    setup.run(setup.path_to_data, setup.test_load)
