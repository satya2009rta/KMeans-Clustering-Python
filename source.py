# DMML Assignment 2 by Arpan Biswas (BMC201604) and Satya Prakash Nayak (BMC201624)
# Importing the required packages
import sys
import random
import math

MAX_ITERATION = 10
K_VALUE = 4
DATASET = "docword.kos.txt"

# Loads the data from file
def loadData(fileName):
    f = open(fileName, "r")
    nDoc = int(f.readline())    # number of document
    nTerm = int(f.readline())   # number of terms
    N = int(f.readline())       # number of input lines to read
    tempDocTerm = [[] for i in range(nDoc)] # list of terms per document along with term frequency
    docTerm = [[] for i in range(nDoc)]     # list of terms per document along with tf-idf value
    docFreq = [0 for i in range(nTerm)]     # document frequency of each term
    for i in range(N):
        # reading the input file
        docID, termID, freq = [int(x) for x in f.readline().split()]
        tempDocTerm[docID-1].append((termID-1, float(freq)))
        docFreq[termID-1] += 1
    f.close()
    # calculating tf-idf value and building docTerm using tempDocTerm
    for j in range(nDoc):
        for (t,f) in tempDocTerm[j]:
            docTerm[j].append((t , f * math.log (float(nDoc)/float(docFreq[t]))))
    return nDoc, nTerm, docTerm

# Returns the norm of a document vector
def norm(dc):
    return math.sqrt(sum([f*f for (i,f) in dc]))

# Calculates the tf-idf distance of two document vector (along with their tf-idf values)
def tf_idfDistance(dc1,dc2):
    i = 0
    j = 0
    dotProduct = 0
    while (i < len(dc1) and j < len(dc2)):
        if dc1[i][0] == dc2[j][0]:
            dotProduct += dc1[i][1]*dc2[j][1]
            i += 1
            j += 1
        elif dc1[i][0] < dc2[j][0]:
            i += 1
        else:
            j += 1
    return 1-dotProduct/(norm(dc1)*norm(dc2))

# Calculate jaccard distance between two document vectors
def jaccardDistance(dc1, dc2):
    doc1 = [i for (i,f) in dc1]
    doc2 = [i for (i,f) in dc2]
    sameWords = set(doc1).intersection(set(doc2))
    differentWords = set(doc1).symmetric_difference(set(doc2))
    dist = 1 - (len(sameWords)/(len(sameWords) + len(differentWords)))
    return dist

# Calculate K means. Here given_seeds is the initial k centroids to start with and distance is the distance function we are using
def kmeans(k, given_seeds, docTerm, nTerm, distance):
    # iterate k means MAX_ITERATION times
    for iteration in range(MAX_ITERATION):
        internal_cluster = {}   # dictionary of centroid with it's set of document cluster
        for j in range(k):
            internal_cluster[given_seeds[j]] = []   # initializing the clusters with empty lists
        # for each document, find the centroid it is closest to and assign it to it's cluster
        for i in range(len(docTerm)):
            minimum = sys.maxsize
            id = 0
            for j in range(k):
                dist = distance(docTerm[given_seeds[j]], docTerm[i])
                if(dist < minimum):
                    minimum = dist
                    id = given_seeds[j]
            internal_cluster[id].append(i)
        # get the new centroids for the clusters we found usign the function findCentroid
        new_seeds = findCentroid(internal_cluster, docTerm, nTerm, distance)
        if(given_seeds == new_seeds):   # if there is no change in centroids, we can stop
            break
        given_seeds = new_seeds
    return internal_cluster

# Find centroids of given clusters
def findCentroid(internal_cluster, docTerm, nTerm, distance):
    new_seeds = []
    for i, docList in internal_cluster.items():         #for each cluster
        numVisitedTerm = [[0,0] for j in range(nTerm)]  # This stores document frequency in cluster and average tf-idf for each term
        for doc in docList:
            for (term,f) in docTerm[doc]:
                numVisitedTerm[term][0] += 1
                numVisitedTerm[term][1] += f/len(docList)
        seed = []
        # We calculate new seed by picking up terms that occur in more than 50% of the documents in the cluster
        for j in range(len(numVisitedTerm)):
            if numVisitedTerm[j][0] > 0.5 * len(docList):
                seed.append((j,numVisitedTerm[j][1]))
        # If the seed has no term (that occur more than 50% in the cluster), we take all the terms that occur in the cluster
        if(len(seed)==0):
            for j in range(len(numVisitedTerm)):
                if numVisitedTerm[j][0] > 0:
                    seed.append((j,numVisitedTerm[j][1]))
        # Finding out the document in the cluster which is nearest to the seed and assigning it the centroid of the cluster
        minimum = sys.maxsize
        id = 0
        for doc in docList:
            dist = distance(docTerm[doc], seed)
            if(dist < minimum):
                minimum = dist
                id = doc
        new_seeds.append(id)
    return new_seeds

# give us initial set of random centroids
def initialSeeds(nDoc, k):
    return random.sample(range(nDoc), k)

# Main Method
def main():
    nDoc, nTerm, docTerm = loadData(DATASET)

    # Calculating and printing result for jaccard distance
    clusters_jaccard = kmeans(K_VALUE, initialSeeds(nDoc, K_VALUE), docTerm, nTerm, jaccardDistance)
    print("For K = ",K_VALUE,", Dataset = ",DATASET," and using jaccard distance :")
    print("DocID of Centroids = ", list(clusters_jaccard.keys()))
    for (i,t) in clusters_jaccard.items():
        print("Cluster for the centroid ", i, " = ", t)
    
    # Calculating and printing result for if-idf distance
    clusters_tf_idf = kmeans(K_VALUE, initialSeeds(nDoc, K_VALUE), docTerm, nTerm, tf_idfDistance)
    print("For K = ",K_VALUE,", Dataset = ",DATASET," and using tf-idf distance :")
    print("DocID of Centroids = ", list(clusters_tf_idf.keys()))
    for (i,t) in clusters_tf_idf.items():
        print("Cluster for the centroid ", i, " = ", t)

# Calling main function
if __name__=="__main__":
    main()
