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
from keras import metrics
from keras.callbacks import History




'''
Relevant data:
data/en.train.json
data/en.dev.json
data/en.test.revdict.json
data/en.trial.complete.json
'''

# default args
args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
args.add_argument('-e','--epochs', default=10, type=int, help='Number of epochs')
args.add_argument('-lr','--learning-rate', default=0.01, type=float, help='Learning rate')
args.add_argument('-do','--dropout', default=0.3, type=float, help='Dropout rate')
args.add_argument('-ea','--early-stopping', default=-1, type=int, help='Early stopping criteria')
args.add_argument('-em','--embedding-size', default=300, type=int, help='Embedding dimension size')
args.add_argument('-et','--embedding-type', default='sgns', type=str, help='Embedding type used: char, sgns, or electra')
args.add_argument('-hs','--hidden-size', default=256, type=int, help='Hidden layer size')
args.add_argument('-T','--train', type=str, help='Train file', required=True)
args.add_argument('-t','--test', type=str, help='Test or Dev file', required=True)
args.add_argument('-b','--batch-size', default=25, type=int, help='Batch Size')
args.add_argument('-s','--save', type=str, help='Name of save file', default='keras_model')
# args.add_argument('-sp','--salt-pepper', default=False, type=boolean, help='Salt and pepper data?')
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
            gloss = line['gloss'].split(' ')
            adjusted_gloss = [START] + gloss + [END]
            data.append(transform_text_sequence(adjusted_gloss))
            # This is the merge lists approach for labels where there are 768 output dimensions
            # Assumption: Each gloss is unique
            #labels.append(line['char'] + line['sgns'] + line['electra'])
            
            # This is the only use one embedding and go from there
            # There are 256 dimensions in each of the three embeddings
            labels.append(line[args.embedding_type])
            
            for tok in adjusted_gloss:
                vocab[tok]+=1
           
    vocab = [w for w in sorted(vocab, key=lambda x:vocab[x], reverse=True)]
    if max_vocab_size:
        vocab = vocab[:max_vocab_size-2]
    vocab = [UNK, PAD] + vocab
    return {k:v for v,k in enumerate(vocab)}, labels, data, labels

def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]


def unvectorize_sequence(seq, vocab):
    translate = sorted(vocab.keys(),key=lambda k:vocab[k])
    return [translate[i] for i in seq]


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
            batch_y.append(label)
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
    print('Label (first 100 dimensions):',gold_labels[0][0:100])
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
    
def load_pretrained_embeddings(glove_file, vocab):
    embedding_matrix = np.zeros((len(vocab), args.embedding_size))
    with open(glove_file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if vocab.get(word) is not None:
                embedding_matrix[vocab[word]] = np.asarray(values[1:], dtype='float32')
            
    embedding_matrix[vocab[UNK]] = np.random.randn(args.embedding_size)
    return embedding_matrix

def transform_text_sequence(seq):
    '''
    Implement this function if you want to transform the input text,
    for example normalizing case.
    Consider replacing punctuation with spaces,
        re.sub("[\(\[].*?[\)\]]", "", definition).replace('  ', ' ')
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
    
    # Using a pre trained embedding has given poor results so far. Definitely seems intuitive to use one though
    
    classifier = keras.Sequential(
        [
            layers.Embedding(input_dim=len(vocab), output_dim=args.embedding_size, weights=[load_pretrained_embeddings('glove.6B/glove.6B.300d.txt', vocab)], trainable=True, name="GloVe_Embedding"),
            layers.LSTM(args.hidden_size, return_sequences=True, name="LSTM_1"),
            layers.Dropout(args.dropout),
            layers.LSTM(args.hidden_size, return_sequences=False, name="LSTM_2"),
            layers.Dropout(args.dropout),
            layers.Dense(256,name="Dense")
        ]
    )

    # ------------------------------------------------------------------------------------------------
    # We are going to try a decaying learning rate
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate= args.learning_rate,decay_steps=1000, decay_rate=0.8)
    
    # Loss is cosine similarity (want to approach -1)
    classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CosineSimilarity(axis=-1,name='cosine_similarity'), metrics=[metrics.mean_squared_error, 'accuracy'])
    
    # Loss is mse (want to approach 0)
    #classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate), loss='mse', metrics=[metrics.mean_squared_error, 'accuracy'])
    
    
    # For model analysis, open the file "data.txt"
    # We will store the evaluation metrics for each epoch in this file
    outputFile = open("evaluation.out","w+")
    outputFile.write("Evaluation Information: \n")
    outputFile.write("Epoch\tLoss\tMSE\tAcc\tSet \n")
    outputFile.write("\n")

    for i in range(args.epochs):
        print('Epoch',i+1,'/',args.epochs)
        # Training
        history = History()
        history = classifier.fit(batch_generator(train_data, train_labels, vocab, labels, batch_size=args.batch_size),
                                 epochs=1, steps_per_epoch=len(train_data)/args.batch_size)
        # Evaluation
        loss, mse, acc = classifier.evaluate(batch_generator(dev_data, dev_labels, vocab, labels),
                                                  steps=len(dev_data))
                                                  
        print('Dev Loss:', loss, 'Dev mse:', mse, 'Dev Acc:', acc)
        
        toWrite1 = str(i) + "\t" + str(history.history['loss']) + "\t" + str(history.history['mean_squared_error']) + "\t" + str(history.history['accuracy']) + "\t" + 'Train' + "\n"
        toWrite2 = str(i) + "\t" + str(loss) + "\t" + str(mse) + "\t" + str(acc) + "\t" + 'Dev/Test' + "\n"
        outputFile.write(toWrite1)
        outputFile.write(toWrite2)
    
    # Output summary of model layers
    classifier.summary()

    classifier.save(args.save)

if __name__ == '__main__':
    main()
