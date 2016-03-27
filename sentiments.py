from sklearn.feature_extraction.text import CountVectorizer
import codecs
from scipy import double,vstack
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class SentimentClassifier:
    def __init__(self, negative='data/SentiWS/SentiWS_v1.8c_Negative.txt',\
        positive='data/SentiWS/SentiWS_v1.8c_Positive.txt'):
        words = dict()
        for line in codecs.open(negative,"r", encoding="utf-8", errors='ignore').readlines():
            parts = line.strip('\n').lower().split('\t')
            words[parts[0].split('|')[0].strip()] = double(parts[1])
            if len(parts)>2:
                for inflection in parts[2].strip('\n').split(','):
                    words[inflection] = double(parts[1])

        for line in codecs.open(positive,"r", encoding="utf-8", errors='ignore').readlines():
            parts = line.strip('\n').lower().split('\t')
            words[parts[0].split('|')[0].strip()] = double(parts[1])
            if len(parts)>2:
                for inflection in parts[2].strip('\n').split(','):
                    words[inflection] = double(parts[1])
        self.bow = CountVectorizer().fit([" ".join(words.keys())])
        data,idx = zip(*[(words[word[0]],(0,word[1])) for word in self.bow.vocabulary_.items()])
        self.words = csr_matrix((data, vstack(idx).T), shape=(1, len(words.keys())))
    
    def predict(self, string):
        if not type(string) is list: string = [string]
        return cosine_similarity(self.words,self.bow.transform(map(lambda x: x.lower().strip(),string)))[0,0]
    
