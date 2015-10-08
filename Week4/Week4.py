import pickle
#import numpy as np
#import matplotlib.pyplot as plt
#import scipy.sparse as spsparse

dirname = r"../Data/"
inputFilename = r"sentences.txt"    
termFilename = r"terms.pkl"
lengthFile = r"lengths.pkl"
defaultSize = 1e4


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
    indexes = indexSentence(words, terms)        
    for i in range(len(words)):
        yield [indexes[l] for l in range(len(words)) if l != i]    

def getHashes(words, terms):
    subSentences = leaveOneOut(words, terms)
    seenHashes = set()    
            
    for ss in subSentences:
        sh = hash(tuple(ss))
       # sh = sh % (2**31 + 1)
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
    print("Single: {0}".format(len(singleHashes)))
    return multiHashes


def loadDocs(fname, size, docs, terms):
    result = dict()
    def func(words):
        id = int(words[0])
        if id in docs:
            result[id] = indexSentence(words[1:], terms)
    forWords(fname, size, func)
    return result

def compWords(d1, d2, tol):
    for w1, w2 in zip(d1, d2):
        if w1 != w2:
            tol -= 1
            if tol < 0:
                return False
    return True

def compDocs(d1, d2):
    ld1 = len(d1)
    ld2 = len(d2)
    lenDiff = abs(ld1 - ld2)
    if lenDiff > 1:
        return False
    if lenDiff == 0:
        # Compare equal length accept 1 diff.
        return compWords(d1, d2, 1)
    
    large = d1
    small = d2
    if ld1 < ld2:
        large = d2
        small = d1
    
    for i in range(len(large)):
        # Leave one out generator
        loo = (large[j] for j in range(len(large)) if j != i)
        if compWords(small, loo, 0):
            return True
    
    # Not within distance 1
    return False

def dups(buckets, docs):
    print("Starting dup count")
    cntP = 0
    cntN = 0
    seenPairs = set()
    matches = set()
    for v in buckets.values():
        def p(l):
            for i in range(len(l)-1):
                for j in range(i+1, len(l)):
                    yield (v[i], v[j])
        for pair in p(v):
            if not pair in seenPairs:                
                seenPairs.add(pair)
                if compDocs(docs[pair[0]], docs[pair[1]]):
                    matches.add(pair)
                    cntP += 1
                else:
                    cntN += 1
    return cntP, cntN, matches

def countFiles(fname, size=defaultSize):
    #storeTerms(inputFilename, termFilename, defaultSize)
    terms = createTerms(fname, size) #loadTerms(termFilename)
    buckets = createBuckets(fname, terms, size)
    print(len(buckets))
    allDocuments = set()
    for v in buckets.values():
        allDocuments.update(v)

    docs = loadDocs(fname, size, allDocuments, terms)
    cntP, cntN, matches = dups(buckets, docs)    
    print("P:{0} N:{1}".format(cntP, cntN))
    return cntP, matches


def getPartitionFilename(fname, partitionLength):
    return "{0}{1}_part{2}.txt".format(dirname, fname, partitionLength)

def partitionFile(fname, size, partitionLength):
    sentences = []
    def func(words):
        sLen = len(words) -1 
        if sLen>= partitionLength and sLen <= partitionLength+1:
            sentences.append("{0}\n".format(" ".join(words)))
    forWords(fname, size, func)
    if len(sentences) <= 0:
        return False
    
    outname = getPartitionFilename(fname, partitionLength)
    with open(outname, "w") as f:        
        f.writelines(sentences)
    return True

def computePartitions(fname, size=defaultSize):
    lengths = dict()

    def f(words):
        l = len(words) - 1
        ul = lengths.setdefault(l, 0)
        lengths[l] = ul + 1
        

    forWords(fname, size, f)
    sortedLengths = sorted([l for l in lengths.keys()])    
    save(lengthFile, sortedLengths)    

    for sl in sortedLengths:
        partitionFile(fname, size, sl)                

computePartitions(inputFilename)
lens = load(lengthFile)

totCount = 0
pMatches = set()
for l in lens:
    fname = getPartitionFilename(inputFilename, l)
    pCount, matches = countFiles(fname)
    pMatches.update(matches)

aCount, aMatches = countFiles(inputFilename)
print("Partition: {0}".format(len(pMatches)))
print("Check: {0}".format(len(aMatches)))


# Up to len 59 use two lengths.
# output all matching pairs, so we can
# remove duplicates, or only count matches for the current partition size.
