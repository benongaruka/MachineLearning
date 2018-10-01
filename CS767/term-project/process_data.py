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

    word_to_int, int_to_word = build_vocab(clean_corpus)    
    return clean_stories, clean_summaries, word_to_int, int_to_word   

def convert_text_to_int(text, word_to_int):
    int_text = []
    for word in text.split():
        int_text.append(word_to_int[word])
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
Input: list of filename 
Output: word_to_int and int_to_word dictionaries
"""
def build_vocab(text):
    
    words = text.split()
    counter = Counter(words)

    pairs = sorted(counter.items(), key=lambda x:(-x[1],x[0]))

    words, c = list(zip(*pairs))

    word_to_int = dict(zip(words, range(len(words))))
    word_to_int['<PAD>'] = len(word_to_int) # Add token to use for padding
    word_to_int['<GO>'] = len(word_to_int) #token needed for training model decoder
    int_to_word = dict(zip(word_to_int.values(), word_to_int.keys()))

    return word_to_int, int_to_word


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

def build_embedding_matrix(word_to_int):
#create embedding_matrix 
    embedding_matrix = np.asarray([len(word_to_int), 300]) # 300 because the trained model has 300 vectors perword
    
    for word, i in word_to_int.items():
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            embedding_matrix[i] = np.array(np.random.uniform(-2.0,1,300)) # give random encoding to word not in index   
    return embedding_matrix         
    
    























