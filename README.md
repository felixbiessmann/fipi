# fipi

A project for political education based on simple machine learning applied to texts of political manifestos, annotated by the political scientists of the [Manifesto Project](https://manifestoproject.wzb.eu/). 

The idea is to use the high-quality (but relatively low volume) manifesto project data annotated by human experts in order to train a text-classification model that can be used to extrapolate the experts' annotations to larger text corpora such as news articles. The hope is to support political education. 

This code is partially based on an [earlier project](https://github.com/kirel/political-affiliation-prediction), which learned a similar text classification model on speeches in the German Parliament. 

## Setup local 

Install [virualenv(-wrapper)](https://virtualenvwrapper.readthedocs.org/en/latest/).
In the folder containing the directory cloned from github then type:

    mkvirtualenv -a fipi fipi

Go to the `web/` folder and  install the dependencies with

    pip install -r requirements.txt

Start the webserver with 
    
    python api.py

A browser window with the app running should have opened. 

## Setup with Docker

Install [Docker](https://docs.docker.com/engine/installation/) and start it. 
In the project root folder then build the docker image and start it with:

    docker-compose up

