import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsparse

dirname = r"../Data/"
inputFilename = r"sentences.txt"    
termFilename = r"terms.pkl"
defaultSize = 1e5


def save(fname, o):
    filePath = dirname+fname
    with open(filePath, "wb") as f:
        pickle.dump(o, f)
def load(fname):
    filePath = dirname+fname
    with open(filePath, "rb") as f:
        return pickle.load(f)


def forWords(fname, size, func):
    filePath = dirname + fname
    count  = 0
    with open(filePath, "r") as f:
        while count < size:
            if (count % 1e5) == 0:
                print("Line: {0}".format(count))
            line = f.readline()
            if not line:
                break
            words = line.rstrip().split(' ')
            func(words)
            count += 1
    
def createTerms(fname, size=defaultSize):
    terms = dict()
    def func(words):
        for w in words:
            terms.setdefault(w, len(terms))
    forWords(fname, size, func)
    return terms

def storeTerms(inputFile, outputFile, size=defaultSize):
    terms = createTerms(inputFile, size)
    save(outputFile, terms)

def loadTerms(fname):
    #storeTerms(inputFilename, "terms.pkl", defaultSize)
    return load(fname)

def indexSentence(s, terms):
    return [terms[w] for w in s]

def leaveOneOut(words, terms):
    indexes = indexSentence(words[1:11], terms)        
    for i in range(10):
        yield [indexes[l] for l in range(10) if l != i]    

def getHashes(words, terms):
    subSentences = leaveOneOut(words, terms)
    seenHashes = set()    
            
    for ss in subSentences:
        sh = hash(tuple(ss))
        # Only deliver unique hashes.
        if sh not in seenHashes:
            seenHashes.add(sh)
            yield sh            


def createBuckets(fname, terms, size=defaultSize):
    singleHashes = dict()
    multiHashes = dict()
    def func(words):
        id = int(words[0])
        allHashes = getHashes(words, terms)        
        for sh in allHashes:
            # Set the value in single hashes if it is not set.
            prevId = singleHashes.setdefault(sh, id)
            # If this is the first time we see sh, 
            # don't update multiHashes.            
            if id != prevId:
                # If this is the second time we see sh
                # we create an entry in multiHashes
                entry = multiHashes.setdefault(sh, [prevId])
                entry.append(id)
    
    forWords(fname, size, func)
    return multiHashes


def loadDocs(fname, size, docs):
    result = dict()
    def func(words):
        id = int(words[0])
        if id in docs:
            result[id] = words
    forWords(fname, size, func)
    return result

def compDocs(d1, d2):
    # Get atual edit distance!


def dups(buckets, docs):
    cntP = 0
    cntN = 0
    seenPairs = set()
    for v in buckets.values():
        def p(l):
            for i in range(len(l)-1):
                for j in range(i+1, len(l)):
                    yield (v[i], v[j])
        for pair in p(v):
            if not pair in seenPairs:
                seenPairs.add(pair)
                if compDocs(docs[pair[0]], docs[pair[1]]):
                    cntP += 1
                else:
                    cntN += 1
    return cntP, cntN

#storeTerms(inputFilename, termFilename, defaultSize)
terms = loadTerms(termFilename)
buckets = createBuckets(inputFilename, terms)
allDocuments = set()
for v in buckets.values():
    allDocuments.update(v)

docs = loadDocs(inputFilename, defaultSize, allDocuments)
cntP, cntN = dups(buckets, docs)    
print("P:{0} N:{1}".format(cntP, cntN))




