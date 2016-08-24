FROM ubuntu:16.04

ENV TERM xterm
ENV DEBIAN_FRONTEND noninteractive

# Not essential, but wise to set the lang
RUN apt-get update && apt-get install -y language-pack-en
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
RUN locale-gen en_US.UTF-8
RUN dpkg-reconfigure locales

# Python binary dependencies, developer tools
RUN apt-get update && apt-get install -y -q \
    vim wget git \
    # Compiler libs
    build-essential make gcc \
    libssl-dev libffi-dev zlib1g-dev \
    # requested for openssl in python
    libffi-dev libicu-dev \
    # Python 3
    python3-dev python3-pip python3-sphinx \
    libzmq3-dev sqlite3 libsqlite3-dev pandoc \
    libcurl4-openssl-dev python3-lxml libpq-dev \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

# Python essential libs
RUN pip3 install --upgrade \
    setuptools pip ipython \
    Flask pandas apscheduler scipy scikit-learn \
    retrying pyicu morfessor polyglot \   
    normality requests dataset \
    git+https://github.com/bundestag/normdatei

RUN git clone http://github.com/abosamoor/pycld2.git; \
    cd pycld2;python3 ./setup.py install

RUN pip3 install readability-lxml

COPY . /app
WORKDIR /app

EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["api.py"]

