FROM continuumio/miniconda
RUN conda config --add channels https://conda.binstar.org/travis \
    && conda config --add channels https://conda.binstar.org/dan_blanchard \
    && conda config --set ssl_verify false \
    && conda update --yes conda
RUN pip install --upgrade pip
COPY . /app
WORKDIR /app
RUN conda install --yes --file conda.txt
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["api.py"]

