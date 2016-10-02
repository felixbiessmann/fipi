from classifier import Classifier
import json,gzip,os,glob

DDIR = "/Users/felix/Data/political-data-science-hackathon/fb-partyposts"

def processAllFacebookData():
    classifier = Classifier(train=False)
    files = glob.glob(os.path.join(DDIR,"*tsv"))
    [processFacebookData(f,classifier) for f in files]

def processFacebookData(fn, classifier):
    nfn = fn[:-4] + "-enriched.json.gzip"
    fh = gzip.open(nfn,'w')
    for post in open(fn).readlines():
        try:
            party,postId,timeStamp,numberLikes,numberComments,\
                _,_,_,text = post.split("\t")
            prediction = classifier.predict(text.strip())
            prediction['party'] = party
            prediction['postId'] = postId
            prediction['timeStamp'] = timeStamp
            prediction['numberLikes'] = numberLikes
            prediction['numberComments'] = numberComments
            fh.write(json.dumps(prediction).encode())
        except:
            Warning("Could not process %s"%post)
            pass
    fh.close()

    
