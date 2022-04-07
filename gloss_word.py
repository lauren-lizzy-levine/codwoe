import argparse
import json
import os, random
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.regularizers import l2

import numpy as np
from keras import backend as K


'''
Relevant data:
data/en.train.json
data/en.dev.json
data/en.test.revdict.json
'''

# default args
args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
args.add_argument('-e','--epochs', default=10, type=int, help='Number of epochs')
args.add_argument('-lr','--learning-rate', default=0.1, type=float, help='Learning rate')
args.add_argument('-do','--dropout', default=0.3, type=float, help='Dropout rate')
args.add_argument('-ea','--early-stopping', default=-1, type=int, help='Early stopping criteria')
args.add_argument('-em','--embedding-size', default=100, type=int, help='Embedding dimension size')
args.add_argument('-hs','--hidden-size', default=10, type=int, help='Hidden layer size')
args.add_argument('-T','--train', type=str, help='Train file', required=True)
args.add_argument('-t','--test', type=str, help='Test or Dev file', required=True)
args.add_argument('-b','--batch-size', default=50, type=int, help='Batch Size')
args = args.parse_args()

train_file = args.train
dev_file = args.test


UNK = '[UNK]'
PAD = '[PAD]'
START = '<s>'
END = '</s>'

# Preprocessing/Data input

def get_vocabulary_and_data(data_file, max_vocab_size=None):
    '''
    This function is called for train, dev, and test splits. Different split, different file names
    
    '''
    vocab = Counter()
    data = []
    labels = []
    with open(data_file, 'r', encoding='utf8') as f:
        dataset = json.load(f)
        for line in dataset:
            if line['gloss'] != '': continue
            adjusted_gloss = [START]
            # This is this part
            #tokens = transform_text_sequence(line.split())
            #for tok in tokens:
            #sent.append(tok)
            #vocab[tok]+=1
            
            gloss = transform_text_sequence(line['gloss'])
            for tokens in gloss:
                adjusted_gloss =
                
        
            
    
    
    
    
    
    #with open(data_file, 'r', encoding='utf8') as f:
      #  for line in f:
        #    print(line[0:100000],"\n"*5)
            
       #     line = line.strip()
       #     if not line: continue
       #     sent = [START]
       
       # This is where they take surname and turn it into tokens
       
       #     tokens = transform_text_sequence(line.split())
       #     for tok in tokens:
       #         sent.append(tok)
       #         vocab[tok]+=1
           # sent.append(END)
           # data.append(sent)
           # vocab[START]+=1
           # vocab[END]+=1
    vocab = [w for w in sorted(vocab, key=lambda x:vocab[x], reverse=True)]
    if max_vocab_size:
        vocab = vocab[:max_vocab_size-2]
    vocab = [UNK, PAD] + vocab

    return {k:v for v,k in enumerate(vocab)}, data


def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]


def unvectorize_sequence(seq, vocab):
    translate = sorted(vocab.keys(),key=lambda k:vocab[k])
    return [translate[i] for i in seq]


def one_hot_encode_label(label, label_set):
    '''
    This function transforms the origin labels ('english', 'czech', 'vietnamese', 'dutch', 'french',...)
    into one hot encoded vectors of length 19 so that when the softmax function turns the NN output layer
    into a probability measure, the one hot encoded vector can be compared to provide an 'error' function.
    '''
    vec = [1.0 if l==label else 0.0 for l in label_set]
    return np.array(vec)


def batch_generator(data, labels, vocab, label_set, batch_size=1):
    '''
    This function takes the input data and labels and outputs lists (batches) with length
    batch_size and padded to be the same length.
    '''
    while True:
        batch_x = []
        batch_y = []
        for doc, label in zip(data,labels):
            batch_x.append(vectorize_sequence(doc, vocab))
            batch_y.append(one_hot_encode_label(label, label_set))
            if len(batch_x) >= batch_size:
                # Pad Sequences in batch to same length
                batch_x = pad_sequences(batch_x, vocab[PAD])
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []


def describe_data(data, gold_labels, label_set, generator):
    batch_x, batch_y = [], []
    for bx, by in generator:
        batch_x = bx
        batch_y = by
        break
    print('Data example:')
    print('Name/sequence:',data[0])
    print('Label:',gold_labels[0])
    print('Label count:', len(label_set))
    print('Data size', len(data))
    print('Batch input shape:', batch_x.shape)
    print('Batch output shape:', batch_y.shape)


def pad_sequences(batch_x, pad_value):
    ''' This function should take a batch of sequences of different lengths
        and pad them with the pad_value token so that they are all the same length.
        Assume that batch_x is a list of lists.
    '''
    pad_length = len(max(batch_x, key=lambda x: len(x)))
    for i, x in enumerate(batch_x):
        if len(x) < pad_length:
            batch_x[i] = x + ([pad_value] * (pad_length - len(x)))
    return batch_x

def transform_text_sequence(seq):
    '''
    Implement this function if you want to transform the input text,
    for example normalizing case.
    '''
    #for i in range(len(seq)):
    #    seq[i] = seq[i].lower()
    return seq


def main():
       
    vocab, labels, train_data, train_labels = get_vocabulary_and_data(train_file)
    _, _, dev_data, dev_labels = get_vocabulary_and_data(dev_file)

    describe_data(train_data, train_labels, labels,
                  batch_generator(train_data, train_labels, vocab, labels, args.batch_size))

    # Implement your model here! ----------------------------------------------------------------------
    # Use the variables args.batch_size, args.hidden_size, args.embedding_size, args.dropout, args.epochs
    # You can input these as command line parameters.
    
    # Print some parameter stats to help orient with data structure
    print("\n")
    print("Hidden size, -hs: ",args.hidden_size)
    print("Embedding size, -em: ",args.embedding_size)
    print("Batch size, -b: ", args.batch_size)
    print("Number of labels: ", len(labels))
    print("Vocab size: ",len(vocab))
    print("\n")
    # Note: This is a character level model so the input_dim for the embedding layer is the vocab/num_chars for the corpra
    classifier = keras.Sequential(
        [
            layers.Embedding(input_dim=len(vocab), output_dim=args.embedding_size,name="Embedding"),
            layers.Bidirectional(LSTM(args.hidden_size, return_sequences=False), name = "LSTM"),
            layers.Dropout(args.dropout, name="Dropout"),
            layers.Dense(len(labels), activation="softmax",name="Dense")
        ]
    )
    
    #Embedding
    #    output: (batch_size, len(vocab), embedding_size)
    #LSTM
    #    output: (batch_size, hidden_size)
    #Dense
    #    output: (batch_size, len(labels))
    # ------------------------------------------------------------------------------------------------

    classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    #classifier.compile(optimizer=keras.optimizers.Adadelta(learning_rate=args.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    for i in range(args.epochs):
        print('Epoch',i+1,'/',args.epochs)
        # Training
        classifier.fit(batch_generator(train_data, train_labels, vocab, labels, batch_size=args.batch_size),
                                 epochs=1, steps_per_epoch=len(train_data)/args.batch_size)
        # Evaluation
        loss, acc = classifier.evaluate(batch_generator(dev_data, dev_labels, vocab, labels),
                                                  steps=len(dev_data))
                                                  
        
        test_loss, test_acc = classifier.evaluate(batch_generator(test_data, test_labels, vocab, labels),
                                                  steps=len(test_data))
        print('Dev Loss:', loss, 'Dev Acc:', acc)
        print('Test Loss:', test_loss, 'Test Acc:', test_acc)
    
    # Output summary of model layers
    classifier.summary()
        
if __name__ == '__main__':
    main()
