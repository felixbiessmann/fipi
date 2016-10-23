# -*- coding: utf8 -*-

from flask.ext.script import Manager
from flask.ext.migrate import Migrate, MigrateCommand
from db_setup import SetupDatabase
from application import app, db
from models import PartyPost, PostLabel, HumanPostLabel


migrate = Migrate(app, db)
manager = Manager(app)

# Initializes the database after backing up the current one if it exists
# Use from command line like 'python manage.py setup_db -p=<path_to_data> '-t=false
manager.add_command('setup_db', SetupDatabase())
# to be used for any potential db changes if needed
manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()
