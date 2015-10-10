import os
import pickle
from time import time
#import numpy as np
#import matplotlib.pyplot as plt
#import scipy.sparse as spsparse

dirname = r"../Data/"
partitionDir = r"{0}partitions/".format(dirname)
inputFilename = r"sentences.txt"    
termFilename = r"terms.pkl"
lengthFile = r"lengths.pkl"
partionDescFile = r"partition_desc.pkl"
defaultSize = 1e6


def save(fname, o):
    filePath = dirname+fname
    with open(filePath, "wb") as f:
        pickle.dump(o, f)
def load(fname):
    filePath = dirname+fname
    with open(filePath, "rb") as f:
        return pickle.load(f)


def forWords(fname, size, func, baseDir = dirname):
    filePath = baseDir + fname
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

def leaveOneOut(indexes):            
    for i in range(len(indexes)):
        yield [indexes[l] for l in range(len(indexes)) if l != i]    

def getHashes(indexes, includeFullSet = True):
    seenHashes = set()
    # Hash the entire index set.
    if includeFullSet:
        sh = hash(tuple(indexes))
        seenHashes.add(sh)
        yield sh
    
    # Generate all subsequences with one item  left out.
    subSet = leaveOneOut(indexes)                    
    for ss in subSet:
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
        indexes = indexSentence(words[1:], terms)
        allHashes = getHashes(indexes)        
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

def compWords2(d1, d2, tol):    
    for w1, w2 in zip(d1, d2):
        if w1 != w2:
            tol -= 1
            if tol < 0:
                return tol
    return tol

def compDocs(d1, d2, baseLen=-1):
    ld1 = len(d1)
    ld2 = len(d2)
    lenDiff = abs(ld1 - ld2)
    if lenDiff > 1:
        return False
    if lenDiff == 0:
        if baseLen > 0 and ld1 != baseLen:
            # Only count equal paris of baseLen
            return False
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

def compDocs2(d1, d2, baseLen=-1):
    ld1 = len(d1)
    ld2 = len(d2)
    lenDiff = abs(ld1 - ld2)
    if lenDiff > 1:
        return -1
    if lenDiff == 0:
        if baseLen > 0 and ld1 != baseLen:
            # Only count equal paris of baseLen
            return -1
        # Compare equal length accept 1 diff.
        return compWords2(d1, d2, 1)
    
    large = d1
    small = d2
    if ld1 < ld2:
        large = d2
        small = d1
    
    for i in range(len(large)):
        # Leave one out generator
        loo = (large[j] for j in range(len(large)) if j != i)
        if compWords2(small, loo, 0) >= 0:
            return 0
    
    # Not within distance 1
    return -1

def dups(buckets, docs, baseLen=-1):
    print("Starting dup count")
    cntP = 0
    cntN = 0
    #seenPairs = set()
    documentToMatches = dict()
    documentDuplicates = dict()
    duplicates = set()
    matches = set()
    for v in buckets.values():
                
        def p(l):
            for i in range(len(l)-1):
                for j in range(i+1, len(l)):
                    yield (v[i], v[j])
        pg = p(v)
        if baseLen > 0:
            # Generate all pairs that should be compared.
            # These include all sentences of base length
            # and pairs (base, sub)
            baseList = []
            subList = []
            for id in v:
                if len(docs[id]) == baseLen:
                    baseList.append(id)
                subList.append(id)
            def pp(b, s):
                for i in range(len(b)):
                    if not b[i] in duplicates:
                        for j in range(i+1, len(s)):
                            if not s[j] in duplicates:
                                yield (b[i], s[j])                            
            pg = pp(baseList, subList)

        for pair in pg:                                                            
            #if compDocs(docs[pair[0]], docs[pair[1]], baseLen):
            cmp = compDocs2(docs[pair[0]], docs[pair[1]], baseLen)
            if cmp == 1:
                entry =  documentDuplicates.get(pair[0], 0)                
                documentDuplicates[pair[0]] = entry +1
                duplicates.add(pair[1])
            elif cmp == 0:
                entry = documentToMatches.setdefault(pair[0], set())
                entry.add(pair[1])                    
            else:
                cntN += 1
    

    # Compute count based on matches and duplicates.
    totalCount = 0
    for k, v in documentToMatches.items():
        otherDup = 0
        for d in v:
            if not d in duplicates:
                otherDup += documentDuplicates.get(d, 0) + 1

        myDup = documentDuplicates.get(k, 0) + 1
        # All duplicates match len(v) documents as well.
        totalCount += myDup*otherDup
    
    for k, v in documentDuplicates.items():
        # Compute within group count.
        myDup = v + 1
        totalCount += myDup*(myDup-1) / 2


    return totalCount, cntN, matches

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
    return "{0}{1}_part{2}.txt".format(partitionDir, fname, partitionLength)

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
    t1 = time()
    if not os.path.exists(partitionDir):
        os.makedirs(partitionDir)
    
    lengths = dict()
    partitionFiles = dict()     
    def f(words):
        l = len(words) - 1
        ul = lengths.setdefault(l, 0)
        lengths[l] = ul + 1
        entry = partitionFiles.get(l, None)
        if not entry:
            pFileName = getPartitionFilename(fname, l)
            pFile = open(pFileName, "w")
            entry = {"name":pFileName, "handle": pFile}
            partitionFiles[l] = entry
        handle = entry["handle"]
        handle.write("{0}\n".format(" ".join(words)))
                                        
    forWords(fname, size, f)
    
    
    sortedLengths = sorted([l for l in partitionFiles.keys()])    
    filenames = []
    for sl in sortedLengths:
        entry = partitionFiles[sl]
        entry["handle"].close()
        filenames.append({"length":sl, "filename":entry["name"]})
    
    save(partionDescFile, filenames)
    t2 = time()
    print("Partitions created in {0} ms".format(t2-t1))

def loadPartition(fname, size=1e7):
    result = []
    def func(words):
        result.append(words)
    forWords(fname, size, func, r"./")
    return result


def concatPartitions(p1, p2):
    for i in p1:
        yield i
    for i in p2:
        yield i

def createPartitionTerms(data):
    terms = dict()
    for words in data:
        for w in words[1:]:
            terms.setdefault(w, len(terms))    
    return terms

def getHashes2(indexes, isBase):
    sub = 1
    if isBase:
        sub = 0
    hLen = int((len(indexes)-sub) / 2)
    


    h1 = hash(tuple(indexes[:hLen]))
    yield h1
    h2 = hash(tuple(indexes[-hLen:]))
    if h1 != h2:
        yield h2

def getPartitionBuckets(data, terms, baseLen):    
    singleHashes = dict()
    multiHashes = dict()
    for words in data:
        id = int(words[0])
        indexes = indexSentence(words[1:], terms)        
        #allHashes = getHashes(indexes, len(indexes) == baseLen)
        allHashes = getHashes2(indexes, len(indexes) == baseLen)
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
    return multiHashes


def getCandidateDictionary(data, docs, terms):
    result = dict()
    for words in data:
        id = int(words[0])
        if id in docs:
            result[id] = indexSentence(words[1:], terms)   
    return result


def countInPartitions(base, sub, baseLen):
    terms = createPartitionTerms(concatPartitions(base, sub))
    buckets = getPartitionBuckets(concatPartitions(base, sub), terms, baseLen)
    print(len(buckets))
    allDocuments = set()
    for v in buckets.values():
        allDocuments.update(v)
    docs = getCandidateDictionary(concatPartitions(base, sub), allDocuments, terms)
    cntP, cntN, matches = dups(buckets, docs, baseLen)    
    print("P:{0} N:{1}".format(cntP, cntN))
    return cntP, cntN, matches



def bar():
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

def partitioned():
    #computePartitions(inputFilename, defaultSize)
    partitions = load(partionDescFile)

    loadedParitions = dict()
    countP = 0
    countN = 0
    allMatces = set()    
    for basePartition, subPartition in zip(partitions[:-1], partitions[1:]):            
    
        print("-- Doing base length {0}".format(basePartition["length"]))

        p1 = loadedParitions.get(basePartition["length"], None)
        if not p1 :
            p1 = loadPartition(basePartition["filename"])
            loadedParitions[basePartition["length"]] = p1
        p2 = loadedParitions.get(subPartition["length"], None)
        if not p2 :
            p2 = loadPartition(subPartition["filename"])
            loadedParitions[subPartition["length"]] = p2
        # Do the stuff.
        cntP, cntN, matches = countInPartitions(p1, p2, basePartition["length"])
        #allMatces.update(matches)
        countP += cntP
        countN += cntN
        # Remove base
        loadedParitions[basePartition["length"]] = None

    # Check the last partition
    # Do the stuff.
    cntP, cntN, matches = countInPartitions(loadedParitions[partitions[-1]["length"]], [], partitions[-1]["length"])
    #allMatces.update(matches)
    countP += cntP
    countN += cntN
    

    print("Part: P{0} N:{1}, Unique: {2}".format(countP, countN, len(allMatces)))
t1 = time()
partitioned()
t2 = time()
print("Total time: {0}".format(t2-t1))
#bar()
#computePartitions(inputFilename, 1e7)



# Count duplicates in the last partition.
# !!!!!!!!!!     count(partitions[-1])

# Up to len 59 use two lengths.
# output all matching pairs, so we can
# remove duplicates, or only count matches for the current partition size.
