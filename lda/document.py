import numpy as np
import codecs
import nltk
import re
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer

class Document():
    
    """ The Doc class rpresents a class of individul documents
    
    """
    
    def __init__(self, post_text, stopwords, clean_length):
        self.text = post_text.lower()
        self.word_list(clean_length, stopwords)
        self.tokens = np.array(wordpunct_tokenize(self.text))
    
    def word_list(self, clean_length, stopwords):
        """
        description: define the word_list attribute (i.e. without stemming)
        """
        self.word_list = np.array(wordpunct_tokenize(self.text))
        self.word_list = np.array([t for t in self.word_list if (t.isalpha() and len(t) > clean_length)])        
        self.word_list = np.array([t for t in self.word_list if t not in stopwords])

    def token_clean(self,length):

        """ 
        description: strip out non-alpha tokens and tokens of length > 'length'
        input: length: cut off length 
        """

        self.tokens = np.array([t for t in self.tokens if (t.isalpha() and len(t) > length)])


    def stopword_remove(self, stopwords):

        """
        description: Remove stopwords from tokens.
        input: stopwords: a suitable list of stopwords
        """

        self.tokens = np.array([t for t in self.tokens if t not in stopwords])


    def stem(self):

        """
        description: Stem tokens with Porter Stemmer.
        """
        
        self.tokens = np.array([PorterStemmer().stem(t) for t in self.tokens])
        
        