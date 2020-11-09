#!/usr/bin/env python
# coding: utf-8

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_colwidth',0)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import re
import random



# To read the input data set
dataset_train=pd.read_csv('/kaggle/input/ift3395-6390-arxiv/train.csv')
test=pd.read_csv('/kaggle/input/ift3395-6390-arxiv/test.csv')
sample_submission=pd.read_csv('/kaggle/input/ift3395-6390-arxiv/sample_submission.csv')

# To print the top 5 records
dataset_train.head()

#List of stop words
stop_words=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


# 1. Do preprocessing
# 2. Build vocab
# 3. build the NB model and fit it
# 

def preprocessing(text):
    text=text.lower() # To make the text lower
    text=text.strip() # To remove the extra space
    text=text.lstrip() #To remove extra space to the left of the text
    text=text.rstrip() # To remove extra space to the right of the text
    text=re.sub('[^a-zA-Z]'," ",text) # To have only characters a to z
    text = re.sub(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"," ",text) # To remove html links
    text=[word for word in text.split() if word not in stop_words and len(word)>=2] # To drop one char word 
                                                                                    #and stop words
    text=' '.join(text)
    return text



#calling the function to preprocess on train and test data 
dataset_train['Abstract']=dataset_train['Abstract'].apply(lambda x:preprocessing(x))
test['Abstract']=test['Abstract'].apply(lambda x:preprocessing(x))
dataset_train.head()

class Vectorizer:
    def __init__(self):
        self.vocab={}
        self.top_vocab=[]
    def build_vocab(self,data):  #To build the vocabulary using the training set
        for sentence in data:
            token=sentence.split()
            for words in token:  #To add the words in the vocab and maintain the occurance
                if words not in self.vocab:  
                    self.vocab[words]=1
                else:
                    self.vocab[words]+=1
        self.top_vocab=sorted(self.vocab,key=self.vocab.get,reverse=True)[0:10000] #To get top 7500 vocab

    def transform(self,data):
        vocab_vec=(self.top_vocab)
        res = np.zeros(shape=(len(data),len(self.top_vocab))) # Creating a dummy np array for final result
        print(res.shape)
        for sentence in range(0,len(data)): 

            text_to_vec=np.zeros(len(self.top_vocab)) # Creating a dummy vector

            for words in data[sentence].split():
                index=-1
                try:
                    index=vocab_vec.index(words) #creating a vector based on the top_vocab size and make it one if present
                    text_to_vec[index]=1
                except ValueError:
                    continue
            res[sentence]=text_to_vec # adding the results as every row and maintaining it in one list

        return res #Return the final transformed vector
    def fit_transform(self,data):
        self.build_vocab(data) #To call the build vocab function
        return self.transform(data) #To call the transform function ie to convert text to meaningful numbers


# Mapping ie. to convert the text to 0-14 based on the category 

categories=['astro-ph','astro-ph.CO','astro-ph.GA','astro-ph.SR','cond-mat.mes-hall','cond-mat.mtrl-sci',
           'cs.LG','gr-qc','hep-ph','hep-th','math.AP','math.CO','physics.optics','physics.optics','stat.ML']

mapping_output={'astro-ph':0,'astro-ph.CO':1,'astro-ph.GA':2,'astro-ph.SR':3,'cond-mat.mes-hall':4,
                'cond-mat.mtrl-sci':5,'cs.LG':6,'gr-qc':7,'hep-ph':8,'hep-th':9,'math.AP':10
                ,'math.CO':11,'physics.optics':12,'quant-ph':13,'stat.ML':14}
dataset_train['Category'].replace(mapping_output, inplace=True)

vec=Vectorizer()
X=vec.fit_transform(dataset_train['Abstract'])


y=dataset_train['Category'].to_numpy()


X_sep_test=vec.transform(test['Abstract'])


#Created a random split function, that splits X & y as train and test
random.seed(30)
def split(X,y,p):
    full_li=list(range(0,int(X.shape[0])))
    random.shuffle(full_li)
    X_train=X[full_li[0:int(p*X.shape[0])]]
    y_train=y[full_li[0:int(p*X.shape[0])]]
    X_test=X[full_li[int(p*X.shape[0]):]]
    y_test=y[full_li[int(p*X.shape[0]):]]
    return X_train,y_train,X_test,y_test


#Calling the split function
X_train,y_train,X_test,y_test=split(X,y,0.7)


# Created a function to print the accuracy and the confusion matrix to analyse the result
y_unique_len=len(list(mapping_output.values()))
def confusion_loss(y_test,predictions):
    matrix=np.zeros((y_unique_len,y_unique_len))
    for true,pred in zip(y_test,predictions):  #creates a confusion matrix
        matrix[int(true)-1][int(pred)-1]+=1

    matrix = matrix.astype(int)
    print(matrix)

    print(np.sum(np.diag(matrix))) # To calculate the correct prediction
    accuracy=np.sum(np.diag(matrix))/len(predictions) #To calculate the accuracy
    return accuracy


class BernoulliNaive:
    def __init(self):
        pass
    def fit(self,X,y):
        #Calculating class wise occurance based on the over all category
        self.n_classes=len(np.unique(y))
        
        self.counts=np.zeros(self.n_classes)
        for i in y:
            self.counts[i]+=1
        self.counts/=len(y)
        
        self.params=np.zeros((self.n_classes,X.shape[1]))
        #nfeatures*nclasses
        for index in range(len(X)):
            self.params[y[index]]+=X[index]
        self.params+=1  # Using Laplace , which is very important since we are finding the product
        class_sums=self.params.sum(axis=1)+self.n_classes
        self.params=self.params/class_sums[:,np.newaxis]
    def predict(self,X):
        neg_prob=np.log(1-self.params)
        res=np.dot(X,(np.log(self.params)-neg_prob).T)
        res+=np.log(self.counts)+neg_prob.sum(axis=1)
        return np.argmax(res,axis=1)

#Call the Bernoulli Class and use the function fit with train data
clf=BernoulliNaive()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test) # to predict based on the training


acc=confusion_loss(y_test,predictions)
print(acc)

clf.fit(X,y)
pred_test=clf.predict(X_sep_test)

cat=list(mapping_output.keys())
res=[cat[i] for i in pred_test]


# To create a submission file for the kaggle
my_submission = pd.DataFrame({'Id': test.Id, 'Category': res})
my_submission.to_csv('submission.csv', index=False)








