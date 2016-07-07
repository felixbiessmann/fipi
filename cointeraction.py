import json,gzip
import pandas as pd
from scipy.sparse import csr_matrix,diags
from scipy.sparse.linalg import eigs,svds
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.externals.joblib import Parallel,delayed
import glob
from multiprocessing import Pool
from sklearn import cross_validation,metrics
import scipy as sp
import os  
import re
import urllib
from bs4 import BeautifulSoup

DDIR = "/Users/felix/Code/Python/fipi/data/cointeractions"

dat = [DDIR + x + ".json.gz" for x in ['afd','npd','pegida']]

urlPat = r'(http://.*\.html)'

def fitKcca(Ks,ncomp=1,gamma=1e-3):
    """
    Fits kernel CCA model
    INPUT:
        Ks       list of kernel matrices (N-by-N)
        ncomp   number of hidden variables
        gamma  regularizer
    """
    N = Ks[0].shape[0]
    m = len(Ks)
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
    norm = adj_mat.sum(axis=0)
    norm[norm == 0] = 1
    return adj_mat / norm

def randomWalkGraphKernelApprox(UA,LA,UB,LB,qA,qB,pA,pB,c=0.5):
    '''
    Computes approximate random walk graph kernel 
    as in algorithm 2 of Kang et al, "Fast Random Walk Graph Kernel", SIAM
    Instead of adjeciency matrix A the algorithm starts with an eigendecomposition
    of A for each graph
    '''
    # compute inverse in eigenspace of product graph
    Lambda = 1./(1./sp.hstack([LB * la for la in LA]) - c)
    L = sp.kron(qA.dot(UA),qB.dot(UB))
    R = sp.kron(UA.T.dot(pA),UB.T.dot(pB))
    return qA.T.dot(pA) * qB.T.dot(pB) + c*L.dot(sp.diag(Lambda)).dot(R)

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

def getCointeractionGraph(fn,maxUsers,numComp,k=3):
    A = readPostWeek(fn,maxUsers)
    timestamp = fn.split("/")[-1].split(".")[0]
    try:
        U,S,V = svds(A,numComp)
        return (S,V.T)
    except:
        return (sp.ones(numComp),A[:numComp,:])

def getCointeractionGraphTuple(x): return getCointeractionGraph(*x)

def graphKernelDummy(A,B):
    return sp.real(A.dot(B.T).sum()).flatten()[0]

def sortDates(x):return int(x.split(".")[0].split("-")[-1])

def getPartyKernel(party,fns,maxUser,numComp, years=['2014','2015','2016']):
    print("Reading %s"%party)
    fns = chain(*map(lambda y: sorted(filter(lambda x: y in x,fns),key=sortDates),years))
    tpls = [(os.path.join(DDIR,party,fn),maxUser,numComp) for fn in fns]
    p = Pool(4)
    cigs = p.map(getCointeractionGraphTuple,tpls)
    N = len(cigs)
    print("Found %d weeks"%N)
    K = sp.zeros((N,N))
    prob = sp.ones(maxUser)/maxUser
    for x in range(N):
        for y in range(x,N):
            K[x,y] = randomWalkGraphKernelApprox(cigs[x][1],cigs[x][0],cigs[y][1],cigs[y][0],prob,prob,prob,prob,c=0.5) 
    #X = sp.sparse.vstack([sp.sparse.hstack([*c[1]]) for c in cigs])
    #K = X.dot(X.T)
    #return sp.array(sp.real(K.todense()))
    return K + K.T

def getPartyKernelTupel(tpl):return getPartyKernel(*tpl)

def readAll(folder=DDIR,numComp=6,years=['2014','2015','2016']):
    fs = [(d,os.listdir(DDIR+"/"+d)) for d in os.listdir(DDIR) if os.path.isdir(DDIR+"/"+d)]
    print("Found %d parties in %s"%(len(fs),folder))
    maxUser = 1+max([max([readMaxUser(os.path.join(DDIR,fss[0],ff)) for ff in fss[1]]) for fss in fs])
    print("Found %d users"%maxUser)
    ptpls = [(p[0],p[1],maxUser,numComp,years) for p in fs]
    return {ptpl[0]:getPartyKernelTupel(ptpl) for ptpl in ptpls}

def evaluateKCCA(Ks,trainIdx,testIdx,numComp,gamma):
    alphas = fit_kcca([k[trainIdx,:][:,trainIdx] for k in Ks.values()],numComp,gamma)

    yhatTrain = [a[0].T.dot(a[1]) for a in zip(alphas,[k[trainIdx,:][:,trainIdx] for k in Ks.values()])]
    yhatTest = [a[0].T.dot(a[1]) for a in zip(alphas,[k[trainIdx,:][:,testIdx] for k in Ks.values()])]
    return yhatTrain,yhatTest

def run_cointeraction(folder=DDIR,numComp=6,years=['2014','2015','2016'],testRatio=.5):
    Ks = readAll(folder,numComp,years)
    N = Ks[list(Ks.keys())[0]].shape[0]    
    trainIdx = range(int(N * (1-testRatio)))
    testIdx = range(int(N * (1-testRatio)),N)
    
    gammas = 10.**sp.arange(-4,4,.1)
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
    plotTrends(Ks,yhatTrain,numComp,'train')
    plotTrends(Ks,yhatTest,numComp,'test')

def getCorrs(yhat,M,ic):
    cors = sp.zeros((M,M))
    for x in range(M):
        for y in range(x+1,M):
            cors[x,y] = sp.corrcoef(yhat[x][ic,:],yhat[y][ic,:])[1,0]
    return cors

def plotTrends(Ks,yhat,numComp,teststr):   
    import pylab as pl
    for ic in range(numComp):
        cors = sp.zeros((len(Ks),len(Ks)))
        for x in range(len(Ks)):
            for y in range(x+1,len(Ks)):
                cors[x,y] = sp.corrcoef(yhat[x][ic,:],yhat[y][ic,:])[1,0]
        pl.figure()    
        pl.imshow(cors.T,interpolation='nearest',cmap='Oranges')
        pl.colorbar()
        pl.yticks(range(len(Ks)),Ks.keys())
        pl.xticks(range(len(Ks)),Ks.keys())
        pl.title("Canonical Correlation %d"%ic)
        pl.savefig("ccs-%d-%s.pdf"%(ic,teststr))
        pl.figure()
        icts = sp.vstack([yhat[x][ic,:] for x in range(len(Ks))])
        pl.plot(icts.T)
        pl.legend(Ks.keys())
        pl.title("Canonical Trend %d"%ic)
        pl.xlabel("Time [weeks]")
        pl.savefig("cc-ts-%d-%s.pdf"%(ic,teststr))
        pl.close('all')
