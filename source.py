# DMML Assignment 2 by Arpan Biswas (BMC201604) and Satya Prakash Nayak (BMC201624)
# Importing the required packages
import numpy as np
import sys
import random

#np.set_printoptions(threshold=np.nan)

def loadData(fileName):
    f = open(fileName, "r")
    nDoc = int(f.readline())
    nTerm = int(f.readline())
    N = int(f.readline())
    docTerm = [[] for i in range(nDoc)]
    for i in range(N):
        docID, termID, freq = [int(x) for x in f.readline().split()]
        docTerm[docID-1].append(termID-1)
    f.close()
    return nDoc, nTerm, docTerm

def jaccardDistance(doc1, doc2):
    sameWords = set(doc1).intersection(set(doc2))
    differentWords = set(doc1).symmetric_difference(set(doc2))
    dist = 1 - (len(sameWords)/(len(sameWords) + len(differentWords)))
    return dist

def findCentroid(internal_cluster, docTerm):
    new_seeds = []
    for i, docList in internal_cluster.items():
        min_sum = sys.maxsize
        id = 0
        for j in range(len(docList)):
            sum_distance = 0
            for k in range(len(docList)):
                sum_distance += jaccardDistance(docTerm[docList[j]], docTerm[docList[k]])
            if(min_sum > sum_distance):
                min_sum = sum_distance
                id = docList[j]
        new_seeds.append(id)
    return new_seeds
            

# Calculate K means
def kmeans(k, given_seeds, docTerm):
    while(True):
        internal_cluster = {}
        for j in range(k):
            internal_cluster[given_seeds[j]] = []
        for i in range(len(docTerm)):
            minimum = sys.maxsize
            id = 0
            for j in range(k):
                dist = jaccardDistance(docTerm[given_seeds[j]], docTerm[i])
                if(dist < minimum):
                    minimum = dist
                    id = given_seeds[j]
            internal_cluster[id].append(i)
        new_seeds = []
        new_seeds = findCentroid(internal_cluster, docTerm)
        if(jaccardDistance(new_seeds, given_seeds)==0):
            break
        given_seeds = new_seeds
    return new_seeds

def initialSeeds(nDoc, k):
    return random.sample(range(nDoc), k)

# Main Method
def main():
    nDoc, nTerm, docTerm = loadData("docword.enron.txt")
    print(kmeans(5, initialSeeds(nDoc, 5), docTerm))


# Calling main function
if __name__=="__main__":
    main()
