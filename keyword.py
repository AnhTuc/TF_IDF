import sys, os
import nltk
import string
import re
import csv
import math
import heapq

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def preprocess(document):
    #PREPROCESS I follow the instruction in page
    # https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/
    
    #converting all letters to lower case
    document=document.lower()

    #removing numbers
    document= re.sub(r'\d+', '',document)

    #removing punctuations
    trans=str.maketrans('', '', string.punctuation)
    document=document.translate(trans)

    #removing white spaces
    document=" ".join(document.split())

    # tokenization
    tokens=word_tokenize(document)

    #removing stop words
    stop_words = set(stopwords.words("english"))
    word_only= [word for word in tokens if word not in stop_words]

    # stemming
    word_in_document=[stemmer.stem(word) for word in word_only]
    return word_in_document

def calculate_idf(corpus):
    idfs={}
    n=len(corpus)
    idfs= dict.fromkeys(corpus[0].keys(),0)

    #Caculate word appear in the all documents
    for i in range(n):
        for word, fre in corpus[i].items():
            if fre>0:
                idfs[word]+=1

    for word, count in idfs.items():
        idfs[word]=math.log10(n/(1+float(count)))
    return idfs

#Return tf of a word in a given document
def calculate_tf(word, words_in_document):
    tf=math.log10(1+words_in_document[word])
    return tf

def find_path(i):
    t=str(i)
    if(i<100):
        if i<10:
            t='00'+str(i)
        else:
            t='0'+str(i)

    return t
#Caculate top_k keywords which have the best tf*idf score
def process(corpus,wordset, top_key=5):
    n=len(corpus)
    top_ks=[]

    #Caculate word frequency in each document
    frequency=[]
    for key in corpus.keys():
        tf={}
        words=dict.fromkeys(wordset,0)
        for word in corpus[key]:
            words[word]+=1
        frequency.append(words)

    idf=calculate_idf(frequency)
    #Caculate top_k word
    i=0
    for key in corpus.keys():
        doc_name=find_path(i+1)+".txt"
        tf_idf={}
        for word in corpus[key]:
            tf=calculate_tf(word,frequency[i])
            tf_idf[word]=tf*float(idf[word])
        
        top_k=heapq.nlargest(top_key, tf_idf, key=tf_idf.get)
        top_k=','.join(top_k)
        top_ks.append([doc_name,top_k])
        i+=1
    return top_ks

def parseInput(dir_path,n_file):
    wordset=set()
    word_in_documents={}
    for i in range(1,n_file+1):
        document=''
        str_name=dir_path+find_path(i)+'.txt'
        with open(str_name,'r') as f:
            for line in f:
                document+=(line.rstrip())
        word_in_documents[i]=preprocess(document)
        wordset=wordset.union(set(word_in_documents[i]))
    
    return wordset,word_in_documents

def write_out(top_k, filename):
    n=len(top_k)
    #header=['filename','top_k']
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for i in range(n):
            writer.writerow(top_k[i])

if __name__ == "__main__":
    dir_path='bbc/tech/'
    outputfile='output.csv'

    if len(sys.argv)>2:
        dir_path=str(sys.argv[1])
        outputfile=str(sys.argv[2])
    
    n_file=0
    for path in os.listdir(dir_path):
    # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            n_file += 1
    
    wordset, word_in_documents = parseInput(dir_path,n_file)
    
    #top_k will have FILENAME as the first element, and the rest are top_k
    top_k=process(word_in_documents,wordset)

    #Write result to csv file
    write_out(top_k,outputfile)
    
   


