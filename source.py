# DMML Assignment 2 by Arpan Biswas (BMC201604) and Satya Prakash Nayak (BMC201624)
# Importing the required packages
import sys
import random


def loadData(fileName):
    f = open(fileName, "r")
    nDoc = int(f.readline())
    nTerm = int(f.readline())
    N = int(f.readline())
    docTerm = [[] for i in range(nDoc)]
    for i in range(N):
        docID, termID, freq = [int(x) for x in f.readline().split()]
        docTerm[docID-1].append((termID-1, freq))
    f.close()
    return nDoc, nTerm, docTerm

def jaccardDistance(dc1, dc2):
    doc1 = [i for (i,f) in dc1]
    doc2 = [i for (i,f) in dc2]
    sameWords = set(doc1).intersection(set(doc2))
    differentWords = set(doc1).symmetric_difference(set(doc2))
    dist = 1 - (len(sameWords)/(len(sameWords) + len(differentWords)))
    return dist

def findCentroid(internal_cluster, docTerm, nTerm, distance):
    new_seeds = []
    for i, docList in internal_cluster.items():
        numVisitedTerm = [[0,0] for j in range(nTerm)]
        for doc in docList:
            for (term,f) in docTerm[doc]:
                numVisitedTerm[term][0] += 1
                numVisitedTerm[term][1] += f/len(docList)
        seed = []
        for j in range(len(numVisitedTerm)):
            if numVisitedTerm[j][0] > 0.5 * len(docList):
                seed.append((j,numVisitedTerm[j][1]))

        minimum = sys.maxsize
        id = 0
        for doc in docList:
            dist = distance(docTerm[doc], seed)
            if(dist < minimum):
                minimum = dist
                id = doc
        new_seeds.append(id)
    return new_seeds
            

# Calculate K means
def kmeans(k, given_seeds, docTerm, nTerm, distance):
    for iteration in range(2):
        internal_cluster = {}
        for j in range(k):
            internal_cluster[given_seeds[j]] = []
        for i in range(len(docTerm)):
            minimum = sys.maxsize
            id = 0
            for j in range(k):
                dist = distance(docTerm[given_seeds[j]], docTerm[i])
                if(dist < minimum):
                    minimum = dist
                    id = given_seeds[j]
            internal_cluster[id].append(i)
        new_seeds = findCentroid(internal_cluster, docTerm, nTerm, distance)
        if(given_seeds == new_seeds):
            break
        given_seeds = new_seeds
    return internal_cluster

def initialSeeds(nDoc, k):
    return random.sample(range(nDoc), k)

# Main Method
def main():
    nDoc, nTerm, docTerm = loadData("docword.kos.txt")
    internal_cluster = kmeans(5, initialSeeds(nDoc, 5), docTerm, nTerm, jaccardDistance)
    for i, t in internal_cluster.items():
        print(i,len(t))


# Calling main function
if __name__=="__main__":
    main()
