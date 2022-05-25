import streamlit as st
import pickle
import numpy as np
import base64
import pyarabic.araby as araby
import re
import string
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import *
model=pickle.load(open('awsatmodelupdate.pkl','rb'))

def clean_text_web(tweets):
#initializing variables 
    clean_text = [] 
    final_sent = ""
    
    tweets = str(tweets).split() #tokenizing 

    for txt in tweets: 

        #removing numbers
        txt = re.sub("[0-9]","",txt) 
        txt = re.sub("[٠-٩]","",txt)

        #removing punctuation marks 
        punc = "["+string.punctuation+"«»،…؟•,ـ\٪“’‘‘”’※"+"]" 
        txt = re.sub(punc," ", txt)     

        #removing English characters 
        txt = re.sub("[A-Za-z]", "", txt) 

        #removing diacritics Arabic (tashkeel)
        txt = araby.strip_tashkeel(txt)

        #removing tatweel (e.g, أبـــشــر)
        txt = araby.strip_tatweel(txt) 

    
        #normalizing some Arabic charachters
        txt = re.sub("[إأٱآا]", "ا", txt) 
        txt = re.sub("ة", "ه", txt)
        txt = re.sub("ى", "ي", txt)
        txt = re.sub("ة", "ه", txt)
        txt = re.sub("گ", "ك", txt)
        
        #Replace @username with empty string
        txt = re.sub('@[^\s]+', ' ', txt)
        
        #Convert www.* or https?://* to " "
        txt = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',txt)
    
    
        #Replace #word with word
        txt = re.sub(r'#([^\s]+)', r'\1', txt)

        # remove repeated letters   
        txt= re.sub(r'(.)\1+', r'\1', txt)

        #rejoining the sentence 
        txt= "".join(ch for ch in txt) 

        for letter in '#.][!XR':
            txt= txt.replace(letter,'')
        
        for letter in 'RT@[A-Za-z0-9]+':
            txt= txt.replace(letter,'') 
        
        
        clean_text.append(txt)
        #removing extra spaces
        final_sent = " ".join([str(x) for x in clean_text]).strip().replace("  "," ")  

    return final_sent

def remove_stopword(text):
    stopwords_list = stopwords.words('arabic')
    tokenizer = RegexpTokenizer(r'\w+')
    text= tokenizer.tokenize(text) 
    for item in text :
        if item not in stopwords_list:
            text=text
    return text

def word_vectorizer_text(text):
    text=pd.Series(text)

    word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 1))
    unigramdataGet= word_vectorizer.transform(text.astype('str'))
    unigramdataGet = unigramdataGet.toarray()

    vocab = word_vectorizer.get_feature_names()
    unigramdata_features=pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
    unigramdata_features[unigramdata_features>0] = 1

    return unigramdata_features

def predict_extremism(textp):
    prediction=model.predict( textp)
    return prediction[0]

def main():
    st.title("Awsat..")
    st.title(" To discover religious extremism. ")
    #texts =['خبر عاجل خلية إرهابية تقتل أطفال']
    texts=st.text_area('here..')##
    Text_clean1 = clean_text_web(texts)##
    Text_clean2=remove_stopword(Text_clean1)##  
    Text_clean3=" ".join(str(x) for x in Text_clean2) 	
    #Text_clean4=word_vectorizer_text(Text_clean2)
    text=pd.Series(Text_clean3)

    word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    ngram_range=(1, 1))
    ts_unigramdataGet22= word_vectorizer.fit_transform(text)

    ts_unigramdataGet= word_vectorizer.transform(text)
    #print(text)
    ts_unigramdataGet = ts_unigramdataGet.toarray()
    vocab = word_vectorizer.get_feature_names()
    #print(vocab)
    ts_unigramdata_features=pd.DataFrame(np.round(ts_unigramdataGet, 1), columns=vocab)
    ts_unigramdata_features[ts_unigramdata_features>0] = 1


    #texts=['خبر عاجل خلية إرهابية تقتل أطفال']
    #pred=LR.predict(x)
    #print (pred)
    #Text_clean3=word_vectorizer_text(Text_clean2)
    if st.button("Predict"):      
        output=predict_extremism(ts_unigramdata_features)
        st.write('The result is : ',output)
        
if __name__=='__main__':
	main()