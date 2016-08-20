from polyglot.mapping import Embedding
from scipy import random
from manifesto_data import api_get_texts
import pickle
import scipy as sp
from sklearn import metrics, cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedKFold

OUT_DIR = "/app/embedding-experiments"
DEFN = "/root/polyglot_data/embeddings2/de/embeddings_pkl.tar.bz2"

def train_embeddings():
    if not os.path.isfile(DEFN):
        downloader.download("embeddings2.de")
    texts = api_get_texts("Germany")
    labels,data = zip(*texts)
    embeddings = Embedding.load(DEFN)
    embeddedText = [sum([embeddings[w] for w in t.split(" ") if w in embeddings]) for t in data]
    labels,embeddedText = zip(*[(l,e) for l,e in zip(labels,embeddedText) if "numpy.ndarray" in str(type(e))])
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(embeddedText, labels, test_size=0.1, random_state=0)
    optimize_hyperparams(X_train,y_train, X_test, y_test)

def randid(N=10):
    return "".join(map(lambda x: str(x),random.randint(0,9,N).tolist()))

def optimize_hyperparams(trainData,trainLabels,evalData,evalLabels, folds=2, idstr='default'):
    text_clf = Pipeline([('clf',LogisticRegression(class_weight='balanced'))])
    parameters = { 'clf__C': (10.**sp.arange(-4,4,2)).tolist()}
    saveId = idstr+"-"+randid()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainData, trainLabels, test_size=0.1, random_state=0)
    # optimize hyperparams on training set
    gs_clf = GridSearchCV(text_clf, parameters, cv=StratifiedKFold(y_train, folds), n_jobs=1,verbose=1)
    # train on training set
    best_clf = gs_clf.fit(X_train, y_train).best_estimator_
    # test on test set
    test_clf = text_clf.set_params(**best_clf.get_params()).fit(X_train,y_train)
    predictedTest = test_clf.predict(X_test)
    # dump report on training held out data with CV
    report = "*** Training Set (CVd) ***\n" + metrics.classification_report(y_test, predictedTest)
    labelsStr = [str(x) for x in test_clf.steps[-1][1].classes_]
    report += '\nConfusion Matrix (rows=true, cols=predicted)\n'+', '.join(labelsStr)+'\n'
    for line in metrics.confusion_matrix(y_test, predictedTest).tolist(): report += str(line)+"\n"
    report += "Accuracy: %0.2f\n"%metrics.accuracy_score(y_test,predictedTest)
    # train again on entire training set
    final_clf = text_clf.set_params(**best_clf.get_params()).fit(trainData, trainLabels)
    # dump classifier to pickle
    pickle.dump(final_clf,open(OUT_DIR+'/pipeline-'+saveId+'.pickle','wb'))
    # test on evaluation data ste
    predictedEval = final_clf.predict(evalData)
    report += "*** Evaluation Set ***\n" + metrics.classification_report(evalLabels, predictedEval)
    labelsStr = [str(x) for x in final_clf.steps[-1][1].classes_]
    report += '\nConfusion Matrix (rows=true, cols=predicted)\n'+', '.join(labelsStr)+'\n'
    for line in metrics.confusion_matrix(evalLabels, predictedEval).tolist(): report += str(line)+"\n"
    report += "Accuracy: %0.2f\n"%metrics.accuracy_score(evalLabels,predictedEval)
    # dump report
    open(OUT_DIR+'/report-'+saveId+'.txt','wb').write(report.encode('utf-8'))
    return report
