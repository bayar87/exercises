""" Groups and Topics

The objective of this task is to explore the structure of the deals.txt file. 

Building on task 1, we now want to start to understand the relationships to help us understand:

1. What groups exist within the deals?
2. What topics exist within the deals?

"""


#***********************************************Clustering deals (documents by groups)*************************************************************

# This code uses the following tunable parameters

# 1. The number of clusters -- kClust
# 2. The number of features for the data -- nFeats

from __future__ import print_function

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np

# We load the data set that we wish to process

###

nFeats =  100
kClust = 2

fcorpus = open('deals.txt')

linelist = []

for x in fcorpus:
	linelist.append(x)

 # vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
 #                                 stop_words='english', use_idf=opts.use_idf)

vectorizer = TfidfVectorizer(max_df=1.0, stop_words='english')
Y = vectorizer.fit_transform(linelist)

lsa = TruncatedSVD(nFeats)

Y =  lsa.fit_transform(Y)

Y = Normalizer(copy=False).fit_transform(Y)

km = KMeans(n_clusters=kClust, init='k-means++', max_iter=100, n_init=1, verbose=True)

km.fit(Y)

# Since we don't have the true labels, to assess the validity of the method and parameters we will use the Silhouette coefficient which is between 0 and 1
#the bigger the coefficient is the better is the performance
#I have commented the print line since the data is very big so it generates an error of memory
#print("Silhouette Coefficient: %0.3f"  % metrics.silhouette_score(Y,km.labels_, metric='euclidean'))

#*************************************************************Topic Extraction using gensim module*******************************

from gensim import corpora, models, similarities

#we build our dictionary from our deals.txt data
dictionary = corpora.Dictionary(line.lower().split() for line in open('deals.txt'))

from nltk.corpus import stopwords

#we define as usual the stop words
stopWords = stopwords.words('english')

#we get the indices of the stop words in our tokenized dictionary
stop_ids = [dictionary.token2id[stopword] for stopword in stopWords
		 	if stopword in dictionary.token2id]


#we get the indices of the words appearing once in our tokenized dictionary
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]

#filtring
dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once

dictionary.compactify() 


class MyCorpus(object):
	def __iter__(self):
		for line in open('deals.txt'):
			yield dictionary.doc2bow(line.lower().split())


corpus = MyCorpus()

#THis is not nec to implement the topic search 

# for vector in corpus:
# 	print(vector)

#we generate the tf-idf model
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]
#This is a parameter to adjust the number of topics that lsi will search for 
nTopics = 2
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=nTopics)
corpus_lsi = lsi[corpus_tfidf]

#Inspection of the lsi object requires logging 
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

for doc in corpus_lsi:
	print(doc)
#print topics: There are topic#0 & topic#1
lsi.print_topics()


#*************************************************************Topic Extraction using NMF********************************************************************** 
#I tried to use the method suggested by Yokoi et al. introduced in their paper titled "Topic Extraction for a Large Document Set with the Topic Integration"

#The NMF method does only work for low dimension since the program generates an error of memory when I tried for our data.
#Moreover, to assess the NMF we need to compute the cophenetic coefficient which requires NMF to be run multiple times for many iterations until it converges

#I have provided the code here:

from time import time
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from sklearn import datasets
nsamples = 1000
nFeats =  1000
nTopics = 2
nTopWords = 20


fcorpus = open('deals.txt')

linelist = []

for x in fcorpus:
	linelist.append(x)


vectorizer = text.CountVectorizer(max_df=1, max_features=nFeats)
counts = vectorizer.fit_transform(linelist[:nsamples])
tfidf = text.TfidfTransformer().fit_transform(counts)



# Now we fit the NMF model
nmf = decomposition.NMF(n_components=nTopics).fit(tfidf)


# Inverse the vectorizer vocabulary to be able
feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-nTopWords - 1:-1]]))
    print()










