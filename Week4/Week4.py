import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as spsparse

filename = r"../Data/sentences.txt"    
size = 1e6
pickledRawFilename = "{0}_{1}k.pkl".format(filename, size/1000)
indexedDataFilename = "{0}_index.pkl".format(filename)
csrDataFilename = "{0}_csr.pkl".format(filename)

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



#pickleCSR()
#D = loadCSR()
#generateSparse(filename, csrDataFilename, 1e5)
d = loadSparse(csrDataFilename)

print(d.nnz)
print(d.shape)

