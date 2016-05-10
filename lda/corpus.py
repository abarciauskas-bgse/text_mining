import numpy as np
import codecs
import nltk
import re
from nltk.tokenize import wordpunct_tokenize
from nltk import PorterStemmer
from collections import Counter
from collections import defaultdict

class Corpus():
    
    """ 
    The Corpus class represents a document collection
     
    """
    def __init__(self, doc_data, stopword_file, clean_length):
        """
        Notice that the __init__ method is invoked everytime an object of the class
        is instantiated
        """
        self.create_stopwords(stopword_file, clean_length)

        #Initialise documents by invoking the appropriate class
        self.docs = [Document(doc, self.stopwords, clean_length) for doc in doc_data] 
        
        self.N = len(self.docs)
        self.clean_length = clean_length
        
        #get a list of stopwords
        self.create_stopwords(stopword_file, clean_length)
        
        #stopword removal, token cleaning and stemming to docs
        self.clean_docs(2)
        
        #create vocabulary
        self.corpus_tokens()
        self.ntotal_tokens = np.sum([len(doc.tokens) for doc in self.docs])
        
    def clean_docs(self, length):
        """ 
        Applies stopword removal, token cleaning and stemming to docs
        """
        for doc in self.docs:
            doc.token_clean(length)
            doc.stopword_remove(self.stopwords)
            doc.stem()        
    
    def create_stopwords(self, stopword_file, length):
        """
        description: parses a file of stowords, removes words of length 'length' and 
        stems it
        input: length: cutoff length for words
               stopword_file: stopwords file to parse
        """
        
        with codecs.open(stopword_file,'r','utf-8') as f: raw = f.read()
        
        self.stopwords = (np.array([PorterStemmer().stem(word) 
                                    for word in list(raw.splitlines()) if len(word) > length]))
        
     
    def corpus_tokens(self):
        """
        description: create a set of all all tokens or in other words a vocabulary
        """
        
        #initialise an empty set
        self.token_set = set()
        for doc in self.docs:
            self.token_set = self.token_set.union(doc.tokens)

    def generate_document_term_matrix(self):
        """
        description: create the document_term_matrix
        """
        dimD = self.N
        # total number of columns
        dimV = len(self.token_set)
        terms_list = list(self.token_set)
        # initialize the matrix
        document_term_matrix = np.zeros((dimD, dimV))        
        for i in range(dimD):
            # count the terms for each document
            document = self.docs[i]
            if i%25==0: print 'counting terms for doc: ' + str(i)
            word_counts = Counter(document.tokens)
            for word_count_pair in word_counts.most_common():
                # split in word and count
                word = word_count_pair[0]
                count = word_count_pair[1]
                # save the term index
                term_idx = terms_list.index(word)
                doc_term_tuple = (i, term_idx)
                document_term_matrix.itemset(doc_term_tuple, count)
        # update the doc_term_matrix attribute        
        self.document_term_matrix = document_term_matrix        
            
    def lda_gibbs(self, K = 3, iters = 10, progress_interval = 1, print_time = False, print_progress = True, print_sanity = False):
        # Initializations
        D = len(corpus.docs)
        V = len(corpus.token_set)
        alpha = 50/K
        eta = 200/V        
        # Randomly assign each word in each document to a topic
        # each array is of "arbitrary" length - length of document's respective word list
        word_topic_assignments = [[]]*corpus.N
        t0 = time.time()
        for doci, doc in enumerate(corpus.docs):
            word_topic_assignments[doci] = []
            for wordi, word in enumerate(doc.tokens):
                word_topic_assignments[doci].append(np.random.randint(0,K))
        t1 = time.time()
        if print_time: print 'Time doing word topic assignments: ' + str(t1 - t0)

        # Initialize document counts of words for each topic, D x K
        document_topic_word_distribution = np.zeros((D, K))
        t0 = time.time()
        for doci, doc in enumerate(corpus.docs):
            # sum the number of words in topic k
            for k in range(0,K):
                document_topic_word_distribution.itemset((doci, k), word_topic_assignments[doci].count(k))
        t1 = time.time()
        if print_time: print 'Time doing document word topic counts: ' + str(t1 - t0)

        # Initialize term topic counts, e.g. number of times term V was allocated to topic K, K x V
        # For every word in every document, determine what it's term index is
        topic_term_distribution = np.zeros((K, V))
        termlist = list(corpus.token_set)
        t0 = time.time()
        for doci, doc in enumerate(corpus.docs):
            for wordi, word in enumerate(doc.tokens):
                termidx = termlist.index(word)
                topic_alloc = word_topic_assignments[doci][wordi]
                current_count = topic_term_distribution.item((topic_alloc,termidx))
                topic_term_distribution.itemset((topic_alloc,termidx), current_count + 1)
        t1 = time.time()
        if print_time: print 'Time doing topic term distribution counts: ' + str(t1 - t0)        

        # Initialize theta - document-specific topic probabilities
        theta = np.zeros((D, K))
        # draw theta - Dirichlet with paramters (alpha + n_{d,k}) - num of words in doc with topic alloc k
        t0 = time.time()
        for doci in range(D):
            theta_params = alpha + document_topic_word_distribution[doci,:]
            # theta for this document
            theta_d = np.random.dirichlet(tuple(theta_params), 1)
            theta[doci,:] = theta_d

        # Initialize beta - x
        beta = np.zeros((K, V))
        # draw beta - Dirichlet with paramters (eta + m_{k,v}) - num of times term appeared in topic k
        for k in range(K):
            beta_params = eta + topic_term_distribution[k,:]
            beta_k = np.random.dirichlet(tuple(beta_params), 1)
            beta[k,:] = beta_k
        t1 = time.time()
        if print_time: print 'Time doing beta and theta initialization distribution counts: ' + str(t1 - t0)    

        if print_sanity:
            terms_to_sanity_check = ['skunk', 'can', 'wine', 'saison']
            for term in terms_to_sanity_check:
                termidx = termlist.index(term)    
                print 'Term: ' + term
                print 'Original topic distribution: ' + str(topic_term_distribution[:,termidx])

        # Iterations
        t0 = time.time()
        for i in range(iters):
            if print_progress and i%progress_interval == 0: print 'Iteration ' + str(i)
            # 1 iteration
            for doci, doc in enumerate(corpus.docs):
                theta_d = theta[doci,:]
                for wordi, word in enumerate(doc.tokens):
                    word_topic_alloc = word_topic_assignments[doci][wordi]
                    termidx = termlist.index(word)
                    beta_v = beta[:,termidx]
                    # decrement counts
                    current_doc_count = document_topic_word_distribution.item((doci, word_topic_alloc))
                    document_topic_word_distribution.itemset((doci, word_topic_alloc), current_doc_count - 1)
                    current_topic_count = topic_term_distribution.item((word_topic_alloc, termidx))
                    topic_term_distribution.itemset((word_topic_alloc, termidx), current_topic_count - 1)
                    probs = theta_d*beta_v/np.sum(theta_d*beta_v)
                    new_topic_alloc = np.random.multinomial(1, tuple(probs))
                    new_topic_alloc = list(new_topic_alloc).index(1)                
                    word_topic_assignments[doci][wordi] = new_topic_alloc
                    # increment counts
                    current_doc_count = document_topic_word_distribution.item((doci, new_topic_alloc))
                    document_topic_word_distribution.itemset((doci, new_topic_alloc), current_doc_count + 1)
                    current_topic_count = topic_term_distribution.item((new_topic_alloc, termidx))
                    topic_term_distribution.itemset((new_topic_alloc, termidx), current_topic_count + 1) 
            # update theta
            for doci in range(D):
                theta_params = alpha + document_topic_word_distribution[doci,:]
                # theta for this document
                theta_d = np.random.dirichlet(tuple(theta_params), 1)
                theta[doci,:] = theta_d
            for k in range(K):
                beta_params = eta + topic_term_distribution[k,:]
                beta_k = np.random.dirichlet(tuple(beta_params), 1)
                beta[k,:] = beta_k
        # end iterations
        t1 = time.time()        
        if print_time: print 'Time spent doing gibbs: ' + str(t1-t0) + ' for ' + str(iters) + ' iterations.'
        if print_sanity:
            for termi, term in enumerate(terms_to_sanity_check):
                termidx = termlist.index(term)
                print 'Term: ' + term
                print 'New topic distribution: ' + str(topic_term_distribution[:,termidx])
        return {'theta': theta, 'beta': beta}

    
