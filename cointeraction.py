import json,gzip
import pandas as pd
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import eigs,svds
from scipy.stats import zscore
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.externals.joblib import Parallel,delayed
import glob
from multiprocessing import Pool
from sklearn import cross_validation,metrics
import pylab as pl
import scipy as sp
import os  
import re
import urllib
from bs4 import BeautifulSoup

DDIR = "/Users/felix/Code/Python/fipi/data/cointeractions"

dat = [DDIR + x + ".json.gz" for x in ['afd','npd','pegida']]

urlPat = r'(http://.*\.html)'

def embed(X,tau):
    '''
    Temporal embedding for scipy.sparse matrices
    '''
    T,D = X.shape
    startInd = max(0,-tau.min())
    stopInd = min(T,T-tau.max())
    Xt = sp.sparse.hstack([X[startInd+t:stopInd+t,:] for t in tau])
    return Xt

def centerKernel(K):
    '''
    Centers a Kernel matrix
    '''
    # K is of dimension N x N
    N = K.shape[0]
    # D is a row vector storing the column averages of K
    D = K.sum(axis=0)/N
    # E is the average of all the entries of K
    E = K.sum() / N
    J = sp.outer(sp.ones(N),D)
    return K - J - J.T + E * sp.ones((N, N))

def fitKcca(Ks,ncomp=1,gamma=1e-3):
    """
    Fits kernel CCA model
    Assumes centered kernel matrices
    INPUT:
        Ks       list of kernel matrices (N-by-N)
        ncomp   number of hidden variables
        gamma  regularizer
    """
    N = Ks[0].shape[0]
    m = len(Ks)
    #Ks = [centerKernel(k) for k in Ks]
    #Ks = [k/sp.linalg.eigh(k)[0].max() for k in Ks]
    # Generate Left-hand side of eigenvalue equation
    VK = sp.vstack(Ks)
    LH = VK.dot(VK.T)
    RH = sp.zeros(LH.shape)
    for ik in range(m):
        # Left-hand side of the eigenvalue equation
        LH[ik*N:(ik+1)*N,ik*N:(ik+1)*N] = 0
        # Right-hand side of the eigenvalue equation
        RH[ik*N:(ik+1)*N,ik*N:(ik+1)*N] = Ks[ik] + sp.eye(N)*gamma
    # Compute the generalized eigenvectors
    c,Vs = sp.linalg.eigh(LH,RH)
    # Sort eigenvectors according to eigenvalues
    Vs = Vs[:,(-c).argsort()]
    alphas = []
    for ik in range(m):
        alphas.append(Vs[ik*N:(ik+1)*N,:ncomp])
    return alphas

def normAdjMat(A):
    '''
    Normalizes columns of adjecency matrix
    '''
    norm = A.sum(axis=0)
    norm[norm == 0] = 1.0
    return A / sp.double(norm)

def testRandomWalk():
    A = normAdjMat(sp.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]]))
    B = normAdjMat(sp.array([[0,1,0,0],[1,0,1,1],[0,1,0,1],[0,1,1,0]]))
    qd = sp.ones(A.shape[0])/A.shape[0]
    k = randomWalkGraphKernel(A,B,qd,qd,qd,qd)    
    LA,UA = sp.linalg.eig(A)
    LB,UB = sp.linalg.eig(B)
    q = csr_matrix(qd).T
    r = 2
    khat = randomWalkGraphKernelApprox(csr_matrix(UA[:,:r]),csr_matrix(LA[:r]),csr_matrix(UB[:,:r]),csr_matrix(LB[:r]),q,q,q,q)
    print("k: %f"%k)
    print("khat: %f"%khat)

def randomWalkGraphKernel(UA,UB,qA,qB,pA,pB,c=0.5):
    '''
    Computes random walk graph kernel 
    '''
    W = sp.kron(normAdjMat(UA),normAdjMat(UB))
    q = sp.kron(qA,qB)
    p = sp.kron(pB,pB)
    return q.dot(sp.linalg.inv(sp.eye(W.shape[0]) - c*W)).dot(p)

def randomWalkGraphKernelApprox(UA,LA,UB,LB,qA,qB,pA,pB,c=0.5):
    '''
    Computes approximate random walk graph kernel 
    as in algorithm 2 of Kang et al, "Fast Random Walk Graph Kernel", SIAM
    Instead of adjeciency matrix A the algorithm starts with an eigendecomposition
    of A for each graph
    '''
    # compute inverse in eigenspace of product graph
    o = csr_matrix(sp.ones(UA.shape[1]*UB.shape[1]))
    Lambda = csr_matrix(o/((o/sp.kron(LA.flatten(),LB.flatten())) - c))
    L = sp.sparse.kron(UA.T.dot(qA),UB.T.dot(qB))
    R = sp.sparse.kron(UA.T.dot(pA),UB.T.dot(pB))
    k = qA.T.dot(pA) * qB.T.dot(pB) + c*Lambda.multiply(R.T).dot(L)
    return k.data[0]

def randomWalkGraphKernelApproxTuple(tpl):
    p = csr_matrix(sp.ones(tpl[0].shape[0])/tpl[0].shape[0]).T
    return (tpl[-2],tpl[-1],randomWalkGraphKernelApprox(tpl[0],tpl[1],tpl[2],tpl[3],p,p,p,p,0.00001))

def readPostLine(line):
    c = line.decode('utf-8').split("\t")
    postId, postType, usrLikes = c[0], c[1], [int(i) for i in c[2:]]
    return postType,postId,usrLikes

def readMaxUser(fn):
    lines = gzip.open(fn).readlines()
    return max(map(lambda x: max(x[2]),map(readPostLine,lines)))

def readPostWeek(fn,maxUsers,numComp=6):
    lines = gzip.open(fn).readlines()
    df = pd.DataFrame(list(map(readPostLine,lines)),columns=['postType','postId','usrLikes'])
    likes = df.groupby("postId")['usrLikes'].agg(sum).values
    rows,cols = zip(*chain(*map(enumerate,likes)))
    return csr_matrix((sp.ones(len(rows)),(rows,cols)),(sp.maximum(len(rows),numComp),maxUsers))

def getUserInteraction(fn,maxUsers):
    A = readPostWeek(fn,maxUsers)
    a = csr_matrix(A.sum(axis=0))
    a.data = a.data/a.data
    ts = "-".join(fn.split(".")[0].split("/")[-1].split("-")[1:])
    return (a,ts)
    
def getUserInteractionTuple(x): return getUserInteraction(*x)

def sortDates(x):return int(x.split(".")[0].split("-")[-1])

def getPartyData(party, fns, maxUser, years=['2014','2015','2016']):
    print("Reading %s"%party)
    fns = chain(*map(lambda y: sorted(filter(lambda x: y in x,fns),key=sortDates),years))
    tpls = [(os.path.join(DDIR,party,fn),maxUser) for fn in fns]
    p = Pool(4)
    data,ts = zip(*p.map(getUserInteractionTuple,tpls))
    return ts,sp.sparse.vstack(data)

def getPartyDataTuple(tpl): return getPartyData(*tpl)

def computeKernel(X):
    return centerKernel(X.dot(X.T).toarray())

def readAll(folder=DDIR,years=['2014','2015','2016']):
    fs = [(d,os.listdir(DDIR+"/"+d)) for d in os.listdir(DDIR) if os.path.isdir(DDIR+"/"+d)]
    print("Found %d parties in %s"%(len(fs),folder))
    maxUser = 1+max([max([readMaxUser(os.path.join(DDIR,fss[0],ff)) for ff in fss[1]]) for fss in fs])
    print("Found %d users"%maxUser)
    ptpls = [(p[0],p[1],maxUser,years) for p in fs]
    data = {ptpl[0]:getPartyDataTuple(ptpl) for ptpl in ptpls}
    return data

def evaluateKCCA(Ks,trainIdx,testIdx,numComp,gamma):
    alphas = fitKcca([k[trainIdx,:][:,trainIdx] for k in Ks],numComp,gamma)

    yhatTrain = [a[0].T.dot(a[1]) for a in zip(alphas,[k[trainIdx,:][:,trainIdx] for k in Ks])]
    yhatTest = [a[0].T.dot(a[1]) for a in zip(alphas,[k[trainIdx,:][:,testIdx] for k in Ks])]
    return yhatTrain,yhatTest

def run_cointeraction(folder=DDIR,numComp=2,years=['2014','2015','2016'],testRatio=.5,tau=sp.array([-2,-1,0])):
    data = readAll(folder,years)
    parties = data.keys()
    N = data[parties[0]].shape[0]
    eData = [embed(data[p][1],tau) for p in parties]
    Ks = [computeKernel(d) for d in eData]
    trainIdx = range(int(N * (1-testRatio)))
    testIdx = range(int(N * (1-testRatio)),N)
    
    gammas = 10.**sp.arange(-5,5,.1)
    cors = []
    for ga in gammas:
        yhatTrain, yhatTest = evaluateKCCA(Ks,trainIdx,testIdx,numComp,ga)        
        cors.append(sum([getCorrs(yhatTest,len(Ks),ic).sum() for ic in range(numComp)]))
    pl.figure()
    pl.plot(cors)
    ticks,labels = zip(*[(x,sp.log10(gammas[x-1])) for x in pl.xticks()[0]])
    pl.xticks(ticks,labels)
    pl.xlabel('log10(gamma)')
    pl.ylabel('summed test correlation')
    pl.savefig("model-selection.pdf")

    yhatTrain, yhatTest = evaluateKCCA(Ks,trainIdx,testIdx,numComp,gammas[sp.argmax(cors)])
    userPatters = [eData[p].T.dot(yhatTrain[p]) for p in range(len(parties))]
    #import pdb;pdb.set_trace()
    #plotTrends(Ks,yhatTrain,userPatterns,'train')
    #plotTrends(Ks,yhatTest,data,'test')

def getCorrs(yhat,M,ic):
    cors = sp.zeros((M,M))
    for x in range(M):
        for y in range(x+1,M):
            cors[x,y] = abs(sp.corrcoef(yhat[x][ic,:],yhat[y][ic,:])[1,0])
    return cors

def plotUserOverlapTimeSeries(x,y,tsLabels,plotStr):
    ovP = []
    for it in range(x.shape[0]):
        ovP.append(len(set(x[it,:].nonzero()[1]).intersection(set(y[it,:].nonzero()[1]))))
    pl.figure(figsize=(10,4))
    pl.plot(ovP)
    xt,_ = pl.xticks()
    pl.xticks(xt,[tsLabels[int(x)] for x in xt[:-1]])
    pl.xlabel('weeks')
    pl.ylabel("overlap")
    pl.title(plotStr)
    pl.savefig(plotStr+".pdf")   

data = readAll()


def plotUserOverlap(data):
    nParties = len(data)
    parties = list(data.keys())
    N,maxUsers = data[parties[0]][1].shape
    # plot correlations of user patterns
    ov = sp.zeros((nParties,nParties))
    for x in range(nParties):
        for y in range(x+1,nParties):
            ovP = []
            px = data[parties[x]][1]
            py = data[parties[y]][1]
            for it in range(N):
                ovP.append(len(set(px[it,:].nonzero()[1]).intersection(set(py[it,:].nonzero()[1]))))
            ov[x,y] = sp.median(ovP)
    pl.figure()    
    pl.imshow(ov.T,interpolation='nearest',cmap='Oranges')
    pl.colorbar()
    pl.yticks(range(nParties),parties)
    pl.xticks(range(nParties),parties)
    pl.title("Median User Overlap")
    pl.savefig("median-user-overlap.pdf") 

def plotTrends(parties,yhat,data,userPatterns,teststr):   
    numComp = yhat.shape[0]
    nParties = len(parties)
    for ic in range(numComp):
        # plot correlations over time of canonical component ic
        # for each pair of parties 
        cors = sp.zeros((nParties,nParties))
        for x in range(nParties):
            for y in range(x+1,nParties):
                cors[x,y] = sp.corrcoef(yhat[x][ic,:],yhat[y][ic,:])[1,0]
        pl.figure()    
        pl.imshow(cors.T,interpolation='nearest',cmap='Oranges')
        pl.colorbar()
        pl.yticks(range(len(Ks)),Ks.keys())
        pl.xticks(range(len(Ks)),Ks.keys())
        pl.title("Canonical Correlation %d"%ic)
        pl.savefig("ccs-%d-%s.pdf"%(ic,teststr))
        
        pl.figure()
        icts = sp.vstack([zscore(yhat[x][ic,:]) for x in range(len(Ks))])
        pl.plot(icts.T)
        pl.legend(Ks.keys())
        pl.title("Canonical Trend %d"%ic)
        pl.xlabel("Time [weeks]")
        pl.savefig("cc-ts-%d-%s.pdf"%(ic,teststr))
        pl.close('all')
