# fipi

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

