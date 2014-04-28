""" Classification

The objective of this task is to build a classifier that can tell us whether a new, unseen deal 
requires a coupon code or not. 

We would like to see a couple of steps:

1. You should use bad_deals.txt and good_deals.txt as training data
2. You should use test_deals.txt to test your code
3. Each time you tune your code, please commit that change so we can see your tuning over time

Also, provide comments on:
    - How general is your classifier?
    - How did you test your classifier?

"""




###########################################################Classification using Naive Bayes###########################################################


from text.classifiers import NaiveBayesClassifier



#Make an image of training data , i.e. good_deal.txt and bad_deals.txt
#This is mainly to not edit our files when assign labels.
with open("good_deals.txt") as f:
     with open("GoodD.txt", "w") as f1:
          for line in f:
              f1.write(line) 


with open("bad_deals.txt") as f:
     with open("BadD.txt", "w") as f1:
          for line in f:
              f1.write(line) 



#assign labels for every deal in both training data.
train = []

with open('GoodD.txt', 'r') as fingood, open('BadD.txt', 'r') as finbad:
    train = ([(deal, 'good_deal') for deal in fingood.readlines()] + [(deal, 'bad_deal') for deal in finbad.readlines()])




#load the testing data
fcorpus = open('test_deals.txt')

test = []

for x in fcorpus:
	test.append(x)


#We classify our data using 
for x in range(len(test)):
    print "The %d^th deal is a %s" % (x,cl.classify(test[x]))

#We don't have the true labels, we can for instance cluster the testing data based on the following characters: %, sale, shop, off, discount etc..
#This is in order to get the true label then classify our data using the NB classifier using the 10-fold Cross Validation.
