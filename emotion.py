# import numpy as np
# X=np.random.randint(2,size=(6,100))
# #print(X)
# Y=([1,2,3,4,4,5])
# from sklearn.naive_bayes import BernoulliNB
# clf=BernoulliNB()
# clf.fit(X,Y)
# #print(clf.predict(X[0:6]))


import nltk
import csv
from nltk.corpus import stopwords
import string
import re #use later to remove numbers and complex punctuations.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import pickle

stop=stopwords.words('english')+list(string.punctuation)
extra=['...']

content=[]
labels=[]
with open("text_emotion.csv","r") as file:
	reader=csv.reader(file)
	i=0
	for row in reader:
		if(row[1]=="neutral" or row[1]=='empty'):
			continue
		else:
			content.append(row[3])
			labels.append(row[1])
			i=i+1
print(i)

content=content[1:30000]
labels=labels[1:30000]
temp_content=[]



for item in content:
	tokens=[i for i in nltk.word_tokenize(item.lower()) if i not in stop and i not in extra]
	temp_content.append(tokens)
#	print(tokens)


#print(temp_content)
fin_content=[]

for item in temp_content:
	fin_content.append(" ".join(item))


# print(fin_content)
# for item in labels:
#	print(item)


# messages = ["Hey hey hey lets go get lunch today",
#            "Did you go home",
#            "Hey I need a favor"]


train_content=fin_content[1:30000]
train_labels=labels[1:30000]

test_content=fin_content[23001:30000]
test_labels=labels[23001:30000]



# messages = ["Hey hey hey lets go get lunch today",
#            "Did you go home",
#            "Hey I need a favor"]




count_vect=CountVectorizer()
X=count_vect.fit_transform(train_content)
print(X.shape)
#print(count_vect.get_feature_names())



# import pandas as pd
# pd_data=pd.DataFrame(X.toarray(), columns=count_vect.get_feature_names())
# print(pd_data)


# tf_idf=TfidfVectorizer()
# X1=tf_idf.fit_transform(train_content)
# print(X.shape)

Y=train_labels #JLT


clf=MultinomialNB()
clf.fit(X,Y)

clf2=SGDClassifier()
clf2.fit(X,Y)




test_X=count_vect.transform(test_content)
print(test_X.shape)

# test_X=tf_idf.transform(test_content)
# print(test_X.shape)


print(clf.score(test_X,test_labels))
print(clf2.score(test_X,test_labels))





pickle.dump(clf,open("NB_clf.sav","wb"))
pickle.dump(clf,open("SVM_clf.sav","wb"))
pickle.dump(count_vect,open("Vectorizer.sav","wb"))
