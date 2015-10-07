import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsparse

filename = r"../Data/sentences.txt"    
size = 1e3
pickledRawFilename = "{0}_{1}k.pkl".format(filename, size/1000)
indexedDataFilename = "{0}_index.pkl".format(filename)
csrDataFilename = "{0}_csr.pkl".format(filename)
bucketFilename = "{0}_bucket.pkl".format(filename)

def loadFile(filename, size, outname):
    result =[]
    with open(filename, "r") as f:
        count = 0
        while count < size:
            line = f.readline()
            if not line:
                break
            words = line.rstrip().split(' ')
            result.append(words)
            count += 1
    
    with open(outname, "wb") as g:
        pickle.dump(result, g)
        
def generatePickle():
    loadFile(filename, size, pickledRawFilename)

def loadPickle():
    outname = "{0}_{1}k.pkl".format(filename, size/1000)
    with open(outname, "rb") as f:
        return pickle.load(f)


def computeIndexes():
    #generatePickle()
    data = loadPickle()
    wordIndexes = dict()
    nextIndex = 0
    result = []
    for idAndWords in data:
        sentenceIndexes = []
        for i in range(len(idAndWords) - 1):
            # Skip the id.
            word = idAndWords[i + 1]
            idx = wordIndexes.get(word)
            if not idx:
                idx = nextIndex
                wordIndexes[word] = nextIndex
                nextIndex += 1
            sentenceIndexes.append(idx)
        result.append(sentenceIndexes)

    return result, wordIndexes

def storeIndexed():
    indexedData, wordIndexes = computeIndexes()
    with open(indexedDataFilename, "wb") as f:
        pickle.dump((indexedData, wordIndexes), f)


def loadIndexed():   
    with open(indexedDataFilename, "rb") as f:
        return pickle.load(f)

def toCSR(sentences):
    indptr = [0]
    indices = []
    data = []
    for s in sentences:
        uniuqeWords = set(s)
        indices.extend(uniuqeWords)
        data.extend([1]*len(uniuqeWords))
        indptr.append(len(indices))
    return spsparse.csr_matrix((data, indices, indptr), dtype=int)

def pickleCSR():
    #storeIndexed()
    #print("Store complete")
    indexedData, wordIndexes = loadIndexed()
    D = toCSR(indexedData)
    with open(csrDataFilename, "wb") as f:
        pickle.dump(D, f)

def loadCSR():
    with open(csrDataFilename, "rb") as f:
        return pickle.load(f)


def generateSparse(filename, outname, size):

    indptr = [0]
    indices = []
    data = []
    wordIndexes = dict()    
    with open(filename, "r") as f:
        count = 0
        while count < size:
            if (count % 1e5) == 0:
                print("Processing {0}".format(count/1e5))
            line = f.readline()
            if not line:
                break
            words = line.rstrip().split(' ')
            uniqueIndices = set()
            for word in words[1:]:
                idx = wordIndexes.setdefault(word, len(wordIndexes))
                uniqueIndices.add(idx)
            indices.extend(uniqueIndices)
            data.extend([1]*len(uniqueIndices))
            indptr.append(len(indices))
            count += 1

    D = spsparse.csr_matrix((data, indices, indptr), dtype=int)
    with open(outname, "wb") as f:
        pickle.dump(D, f)
    
def loadSparse(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)




def getWordHashes(wordArray):
    return [hash(w) for w in wordArray[1:11]]


def generateBucketCandidates(filename, size, prefixSize=10):
    cnt = dict()
    ids = set()
    with open(filename, "r") as f:
        count = 0
        while count < size:
            if (count % 1e5) == 0:
                print("Processing {0}".format(count/1e5))
            line = f.readline()
            if not line:
                break
            words = line.rstrip().split(' ')
                        
            sLen = len(words)
            # We now that at the sentences are at least length 10.
            # Hash the tuple of all words except i for each i 0-9.
            wordHashes = getWordHashes(words)
            for i in range(prefixSize):
                t = hash(tuple([wordHashes[l] for l in range(prefixSize) if l != i]))
                e = cnt.get(t, False)
                if e:
                    ids.add(t)
                else:
                    cnt[t] = True
            count += 1
    return ids

def storeBucketCandidates():
    buckets = generateBucketCandidates(filename, size, prefixSize=10)
    with open(bucketFilename, "wb") as f:
        pickle.dump(buckets, f)        

def loadBuckets():
    #storeBucketCandidates()
    with open(bucketFilename, "rb") as f:
        return pickle.load(f)

def indexSentence(terms, words):
    result = []
    for w in words[1:]:
        idx = terms.setdefault(w, len(terms))
        result.append(idx)
    return result

def count(filename, size, prefixSize=10):
    buckets = generateBucketCandidates(filename, size, prefixSize)#loadBuckets()
    print("Loaded buckets")
    terms = dict()
    result = dict()
    with open(filename, "r") as f:
        count = 0
        while count < size:
            if (count % 1e5) == 0:
                print("Processing {0}".format(count/1e5))
            line = f.readline()
            if not line:
                break
            words = line.rstrip().split(' ')
                        
            sLen = len(words)
            # We now that at the sentences are at least length 10.
            # Hash the tuple of all words except i for each i 0-9.
            wordHashes = getWordHashes(words)
            indexedSentence = None
            for i in range(prefixSize):
                t = hash(tuple([wordHashes[l] for l in range(prefixSize) if l != i]))
                if t in buckets:
                    #print("found bucket")
                    # We should do stuff for this bucket.
                    if not indexedSentence:
                        indexedSentence = indexSentence(terms, words)
                    entry = result.get(t, None)
                    if not entry:
                        entry = []
                        result[t] = entry
                    entry.append({"i":int(words[0]), "s":words[1:]})
            count += 1
    
    return result

#Second pass
#d = count(filename, size)
#print(len(d))




