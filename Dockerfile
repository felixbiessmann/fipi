FROM continuumio/miniconda
RUN conda update --yes conda
RUN pip install --upgrade pip
COPY . /app
WORKDIR /app
RUN cat requirements.txt | grep 'scipy\|numpy\|^lxml\|scikit-learn\|pandas' > conda.txt
RUN conda install --yes --file conda.txt
RUN pip install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["api.py"]

