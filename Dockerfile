FROM resin/rpi-raspbian

# Python binary dependencies, developer tools
RUN apt-get update && apt-get install -y -q \
    vim wget git \
    # Compiler libs
    build-essential make gcc \
    libssl-dev libffi-dev zlib1g-dev \
    libatlas-base-dev gfortran \
    libffi-dev libicu-dev \
    sqlite3 libsqlite3-dev \
    libcurl4-openssl-dev \
    libxml2-dev libxslt1-dev \
    # Python 3
    python3-dev python3-pip \ 
    python3-numpy python3-scipy \
    python3-lxml python3-bs4 python3-pandas \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

# Python essential libs
RUN pip3 install --upgrade \
    setuptools pip ipython lxml \
    Flask apscheduler scikit-learn \
    retrying requests readability-lxml 
    #\  
    #normality dataset \
    #git+https://github.com/bundestag/normdatei

COPY . /app
WORKDIR /app

EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["api.py"]

