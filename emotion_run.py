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


nb_clf=pickle.load(open("NB_clf.sav","rb"))
svm_clf=pickle.load(open("SVM_clf.sav","rb"))
dt_clf=pickle.load(open("DT_clf.sav","rb"))
#We'll be using SVM as the accuracy is better.

count_vect=pickle.load(open("Vectorizer.sav","rb"))
print("loaded")


input_user=input()
sentence=[]
sentence.append(input_user)

print(sentence)

X=count_vect.transform(sentence)
res0=nb_clf.predict(X)
res1=svm_clf.predict(X)
res2=dt_clf.predict(X)



if(res0=="worry" or res0=="sadness" or res0=="hate"):
	result='negative emotion'
else:
 	result='positive emotion'

if(res1=="worry" or res1=="sadness" or res1=="hate"):
	result='negative emotion'
else:
	result='positive emotion'

if(res2=="worry" or res2=="sadness" or res2=="hate"):
	result='negative emotion'
else:
	result='positive emotion'

print(res0)
print(res1)
print(res2)