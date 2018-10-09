import sys 
from pickle import dump, load
import string
import os
import argparse
from os import listdir
import numpy as np
from collections import Counter
#This module provides helper functions for processing files for NN input

#read a file and return text
def load_file(filename, read_by_line = False):
    file = open(filename, encoding='utf-8')
    if read_by_line == True :
        content = file.readlines()
    else:
        content = file.read()
    file.close()
    return content  

#padding for model input 
def pad_sentence_batch(sentence_batch, word_to_int):
    max_sentence_length = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [word_to_int['<PAD>']]*(max_sentence_length - len(sentence)) for sentence in sentence_batch]


#load num stories in a directory
def load_stories(directory, num=10):
    stories = list()
    print("Directory: " + directory)
    print("Number of files to process: " + str(num))

    #Full text for building vocabulary
    clean_corpus = "" 


    #variables
    clean_stories= []
    clean_summaries = []

    for f in listdir(directory)[:num]:
        print("Processing file: "+f) 
        filename = os.path.join(directory,f) 
        doc = load_file(filename)
        story, highlights = split_story(doc)
        story = clean_text(story)
        highlights = clean_text(highlights)
        clean_corpus += ' ' + story + ' ' + highlights
        clean_stories.append(story)
        clean_summaries.append(highlights)

        word_count = get_word_count(clean_corpus)    
    return clean_stories, clean_summaries, word_count    
'''
input is array of texts. each element is a document 
output array of each text converted to an array of ints
'''
def convert_text_to_int(text, word_to_int, eos = False):
    int_text = []

    for sentence in text:
        int_sentence = []
        for word in sentence.split():
            if word in word_to_int:
                int_sentence.append(word_to_int[word])
            else:
                 int_sentence.append(word_to_int['<UNK>'])
        
        if eos:
            int_sentence.append(word_to_int['<EOS>'])
        
        int_text.append(int_sentence)
    return int_text    

#split a story into story and summary
#each doc has a story and highlights tagged by @highlight token
def split_story(doc):
    index = doc.find('@highlight')
    story, highlights = doc[:index], doc[index:].split('@highlight')
    highlights = [h.strip() for h in highlights if len(highlights)>0]
    highlights = " ".join(highlights)
    return story, highlights

#Clean the data
def clean_text(text):
    trans_table = str.maketrans('','',string.punctuation)
    text = text.lower()
    index = text.find('CNN --')
    if index > -1:
        text = text[index+len('CNN'):]
    words  = text.split()
    words = [word.translate(trans_table) for word in words] #remove punctuation
    words = [word for word in words if word.isalpha()]
    #remove empty strings
    cleaned = [c for c in words if len(c) > 0]
    cleaned_text = ' '.join(cleaned)
    
    return cleaned_text

#Create embedding matrix 
#Based pre-trained model: ConceptNet NumberBatch https://github.com/commonsense/conceptnet-numberbatch
"""
Input: text corpus 
Output:word_count 
"""
def get_word_count(text):
    
    words = text.split()
    counter = Counter(words)


    pairs = sorted(counter.items(), key=lambda x:(-x[1],x[0]))

    words, c = list(zip(*pairs))

    word_count = dict(zip(words, range(len(words))))

    return word_count 


"""
creates an embedding index dictionary from CN trained model
output: dictionary embedding_index of words and corresponding 300 dim vectors
"""
def init_embedding_index(embedding_file = None):
    embedding_index = {}
    if embedding_file == None:
        CN_path = './numberbatch-en.txt'
    
    with open(CN_path) as f:
        for line in f:
            values = line.split()
            word = values[0] #first item is the word, followed by 300 dimension encoding
            embedding_index[word] = np.asarray(values[1:], dtype='float32')
    return embedding_index        


"""
Input: word_count, threshold 
output: word_to_int, int_to_word, embedding_matrix
"""
def build_vocab(word_count, threshold=10):
#create embedding_matrix 
    
    print('Initializing embedding matrix.')
    embedding_index = init_embedding_index()
    print('embedding matrix initialized')    
    #create word_to_int 
    special_tokens = ['<EOS>','<PAD>','<GO>','<UNK>']
    word_to_int = {}

    for token in special_tokens:
        word_to_int[token]=len(word_to_int)
    
    print('Creating word_to_int')
    for word, c in word_count.items():
        if c > threshold or word in embedding_index:
            word_to_int[word] = len(word_to_int)
            
            
    embedding_matrix = np.zeros((len(word_to_int), 300),dtype=np.float32) # 300 because the trained model has 300 vectors perword

    for word, i in word_to_int.items():
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            embedding_vector = np.array(np.random.uniform(-2.0,1,300)) # give random encoding to word not in index
            embedding_matrix[i] = embedding_vector    
            embedding_index[word] = embedding_vector

    int_to_word = dict(zip(word_to_int.values(), word_to_int.keys()))        
    return embedding_matrix, word_to_int, int_to_word         
    
    























