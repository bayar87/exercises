""" Features

The objective of this task is to explore the corpus, deals.txt. 

The deals.txt file is a collection of deal descriptions, separated by a new line, from which 
we want to glean the following insights:

1. What is the most popular term across all the deals?
2. What is the least popular term across all the deals?
3. How many types of guitars are mentioned across all the deals?

"""

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

#import a list of stop words
from nltk.corpus import stopwords

stopWords = stopwords.words('english')


#Open the file deals.txt
fcorpus = open('deals.txt')
linelist = []

#the x^{th} element of linelist set is the x^th line in data.txt
for x in fcorpus:
	linelist.append(x)

#remove the stop words e.g. at, and, etc.., from our data and extract the features (occurrence frequency of every word) from all the documents
#we consider every line (deal) as a document it self 
vectorizer = CountVectorizer(min_df=1, stop_words = stopWords)

X = vectorizer.fit_transform(linelist)
#The rows of the matrix X are the documents (deals) and the columns are the features which are the occurrence frequencies of word
#More specifically, X(i,j) is the occurrence frequency of the j^{th} word in the i{th} document
#Now we sum up over features (columns of X) to find the total occurrence frequency of every j^{th} word in the entire deals.txt file
#then we store that in a row vector totFreq
totFreq = X.sum(2)

#get the index of the most frequent and the least frequent word in the entire deal.txt file
maxInd = totFreq.argmax()
minInd = totFreq.argmin()

#From the indices we can retrieve the feature name, i.e. the word itself
featList = vectorizer.get_feature_names()

#Now we print the results for the most frequent and least frequent word 
print featList.pop(maxInd), 'is repeated' ,   totFreq[0,maxInd], 'times'
featList = vectorizer.get_feature_names()
print featList.pop(minInd), 'is repeated' ,   totFreq[0,minInd], 'times'



# Now we try to find the number of bigrams with guitar as the second words
#Import the module as usual
import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()

from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords

#load the stop words as usual
stopWords = stopwords.words('english')



# We load our entire file and read through it as one set of characters
rdr = PlaintextCorpusReader('.','deals.txt')
#set two filters for our bigrams
#1st filter we need the word guitar to be as the second word in every bigram
#2nd filter we need our bigrams to be diffrent from the stop words
filt = lambda w1, w2: 'uitar' not in w2
filtStp = lambda w1, w2: (w1 or w2 ) in stopwords.words('english')
#Extra Filter
ignore_list = ['Shop', 'shop', 'Online', 'online', 'Fingerstyle', 'Learn', 'GuitarCenter', 'at', 'on', 'Arts', 'for', '!', 'the', '-', 'Boutique', '4x12', '7', '.', '(', ')', ',', 'from', 'com', 'four', '2x12', '"', 'Flatpicking', 'Category', 'Destinations', 'HD', 'Flatpick', 'Products', 'Drums', 'Series']
filtExtr = filtStp = lambda w1, w2: (w1 or w2 ) in ignore_list
## find bigrams
finder = BigramCollocationFinder.from_words(rdr.words())

finder.apply_ngram_filter(filt)
finder.apply_ngram_filter(filtStp)
finder.apply_ngram_filter(filtExtr)
#compute the number of occurrences of bigrams
scored = finder.score_ngrams(bigram_measures.raw_freq)

#sort the bigrams by their occurrence frequencies
sorted(bigram for bigram, score in scored)

unique_Guitar =  list(set(sorted(bigram for bigram, score in scored)))
print "The number of different guitars in all the deals is %d" % len(unique_Guitar)

# return the 10 n-grams with the highest PMI
print finder.nbest(bigram_measures.likelihood_ratio, 10)
