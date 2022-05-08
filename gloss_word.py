import argparse
import json
import os, random
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, GaussianNoise
from keras.regularizers import l2

import numpy as np
from keras import backend as K
from keras import metrics
from keras.callbacks import History
from preproc import stem, lemmatize, lower, rem_stop, rem_punc

# Note difference. Model is now in fuction. Made training easier. Also evaluation.out is append opened so delete localy as needed but dont forget. Added code to get started on word/noun/pos interpretation
'''
Relevant data:
data/en.train.json
data/en.dev.json
data/en.trial.complete.json
data/en.test.revdict.json

Optimized hyper parameters (strafied based on embedding)
SGNS:    -e=1  -lr=0.001  -do=0.1  -hs=20  -b=5   -gn=0.005  LOSS:0.31288  MSE:0.97931  ACC:0.04471
CHAR:    -e=2  -lr=0.001  -do=0.1  -hs=5   -b=20  -gn=0.005  LOSS:0.79997  MSE:0.36708  ACC:0.50369
ELECTRA: -e=1  -lr=0.01   -do=0.2  -hs=50  -b=20  -gn=0.005  LOSS:0.84549  MSE:1.33066  ACC:0.56502
'''

# default args
args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
args.add_argument('-e','--epochs', default=10, type=int, help='Number of epochs')
args.add_argument('-lr','--learning-rate', default=0.001, type=float, help='Learning rate')
args.add_argument('-do','--dropout', default=0.3, type=float, help='Dropout rate')
args.add_argument('-ea','--early-stopping', default=-1, type=int, help='Early stopping criteria')
args.add_argument('-em','--embedding-size', default=300, type=int, help='Embedding dimension size')
args.add_argument('-et','--embedding-type', default='sgns', type=str, help='Embedding type used: char, sgns, or electra')
args.add_argument('-hs','--hidden-size', default=256, type=int, help='Hidden layer size')
args.add_argument('-T','--train', type=str, help='Train file', required=True)
args.add_argument('-t','--test', type=str, help='Test or Dev file', required=True)
args.add_argument('-b','--batch-size', default=25, type=float, help='Batch Size')
args.add_argument('-s','--save', type=str, help='Name of save file', default='keras_mode')
args.add_argument('-gn','--gaussian-noise', default=0,type=float, help='STDDEV for gaussian noise. None or 0 is default')
args.add_argument('-pp','--preprocessing', type=str, help='Preprocess the gloss', default='None')
## lem, stem, lower, stop, punc
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
    
    # Be able to pull additional data from records if using the Trail data set
    #if (data_file[len(data_file) -13:] == 'complete.json'):
    #
    #else:
    # ...
    
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
    res = seq[1:-1]
    if 'stem' in args.preprocessing:
        res = stem(res)
    if 'lem' in args.preprocessing:
        res = lemmatize(res)
    if 'stop' in args.preprocessing:
        res = rem_stop(res)
    if 'punc' in args.preprocessing:
        res = rem_punc(res)
    if 'lower' in args.preprocessing:
        res = lower(res)
    res.insert(0,'<s>')
    res.append('</s>')
    return res

def model(train_data, train_labels,dev_data, dev_labels, vocab, labels, params):
    
    # Print some parameter stats to help orient with data structure
    print("\n")
    print("Hidden size, -hs: ",params.hidden_size)
    print("Embedding size, -em: ",params.embedding_size)
    print("Learn rate, -lr: ",params.learning_rate)
    print("Dropout, -do: ", params.dropout)
    print("Batch size, -b: ", params.batch_size)
    print("Gauss, -gn: ", params.gaussian_noise)
    print("Number of labels: ", len(labels))
    print("Vocab size: ",len(vocab))
    print("\n")
    
    # Using a pre trained embedding has given poor results so far. Definitely seems intuitive to use one though
    classifier = keras.Sequential(
        [
            layers.Embedding(input_dim=len(vocab), output_dim=params.embedding_size, weights=[load_pretrained_embeddings('glove.6B/glove.6B.300d.txt', vocab)], trainable=True),
            layers.GaussianNoise(params.gaussian_noise),
            layers.Bidirectional(LSTM(params.hidden_size, return_sequences=True, name="LSTM_1")),
            layers.Dropout(params.dropout),
            layers.LSTM(params.hidden_size, return_sequences=False, name="LSTM_2"),
            layers.Dropout(params.dropout),
            layers.Dense(256,name="Dense")
        ]
    )
    
    # We are going to try a decaying learning rate
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=
                params.learning_rate,decay_steps=10000, decay_rate=0.9)
    
    # Loss is cosine similarity (want to approach -1)
    classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CosineSimilarity(axis=-1,name='cosine_similarity'), metrics=[metrics.mean_squared_error, 'accuracy'])
    
    # Loss is mse (want to approach 0)
    #classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate), loss='mse', metrics=[metrics.mean_squared_error, 'accuracy'])
    
    # For model analysis, open the file "data.txt"
    # We will store the evaluation metrics for each epoch in this file
    outputFile = open("evaluation.out","a")
    #outputFile.write("Evaluation Information: \n")
    #outputFile.write(str("Epoch"+"\t"*2+"lr"+"\t"*2+"do"+"\t"*2+"hs"+"\t"*2+"b"+"\t"*2+"gn"+"\t"*2+"Loss"+"\t"*2+"MSE"+"\t"*2+"Acc"+"\t"*2+"Set"+"\n"))
    #outputFile.write("\n")


    for i in range(args.epochs):
        print('Epoch',i+1,'/',args.epochs)
        
        # Training
        history = History()
        history = classifier.fit(batch_generator(train_data, train_labels, vocab, labels, batch_size=params.batch_size),
                                 epochs=1, steps_per_epoch=len(train_data)/params.batch_size)
        # Evaluation
        loss, mse, acc = classifier.evaluate(batch_generator(dev_data, dev_labels, vocab, labels),
                                                  steps=len(dev_data))
                                                  
        print('Dev Loss:', loss, 'Dev mse:', mse, 'Dev Acc:', acc)
        
    
        toWrite1 = '%s\t\t%.3f\t\t%.3f\t\t%f\t%f\t%.3f\t\t%.5f\t%.5f\t\t%.5f  \t %s\n' % (str(i+1), params.learning_rate, params.dropout, params.hidden_size, params.batch_size, params.gaussian_noise, history.history['loss'][0], history.history['mean_squared_error'][0],history.history['accuracy'][0],"Train")
        toWrite2 = '%s\t\t%.3f\t\t%.3f\t\t%f\t%f\t%.3f\t\t%.5f\t%.5f\t\t%.5f  \t %s\n' % (str(i+1), params.learning_rate, params.dropout, params.hidden_size, params.batch_size, params.gaussian_noise, loss, mse, acc,"Dev/Test")
    
        outputFile.write(toWrite1)
        outputFile.write(toWrite2)
       
    # Save Model
    classifier.save(args.save)
    
    return classifier, history
    
def main():
    #with open("data/en.trial.complete.json", 'r', encoding='utf8') as f:
    #    dataset = json.load(f)
     #   for line in dataset:
     #       print(line)
    
    vocab, labels, train_data, train_labels = get_vocabulary_and_data(train_file)
    _, _, dev_data, dev_labels = get_vocabulary_and_data(dev_file)

    describe_data(train_data, train_labels, labels,
                  batch_generator(train_data, train_labels, vocab, labels, args.batch_size))
    
    classifier, history = model(train_data, train_labels,dev_data, dev_labels, vocab, labels, args)
    
    # ---------------------------------------------------------------- #
    #                   Hyper parameter tuning  Phase I                #
    # -----------------------------------------------------------------#
    #x = [0.1,0.5,1,2,5,10]
    #parameter_space = {'lr': [0.001*i for i in x],'do':[0.1*i for i in x],'hs':[100*i for i in x],
    #    'b':[10*i for i in x],'gn':[0.05*i for i in x]}
    #print(parameter_space)
    
    #random_index = random.randint(0,4)
    #for i in range (25):
    #    args.learning_rate = parameter_space['lr'][random.randint(0,4)]
    #    args.dropout = parameter_space['do'][random.randint(0,4)]
    #    args.hidden_size = int(parameter_space['hs'][random.randint(0,4)])
    #    args.batch_size = parameter_space['b'][random.randint(0,4)]
    #    args.gaussian_noise = parameter_space['gn'][random.randint(0,4)]
    #    classifier, history = model(train_data, train_labels,dev_data, dev_labels, vocab, labels, args)
        
    # ---------------------------------------------------------------- #
    #                   Hyper parameter tuning  Phase II               #
    # -----------------------------------------------------------------#
    #x = [1,2,5,10]
    #y = [1,2,5]
    #z = [1,2]
    
    #parameter_space = {'lr': [0.002*i for i in y],'do':[0,0.1,0.2],'hs':[5*i for i in x],
    #    'b':[10*i for i in z],'gn':[0.05]}
    #print(parameter_space)
    
    #for a in parameter_space['lr']:
    #    for b in parameter_space['do']:
    #        for c in parameter_space['hs']:
    #            for d in parameter_space['b']:
    #                args.learning_rate = a
    #                args.dropout = b
    #                args.hidden_size = c
    #                args.batch_size = d
    #                args.gaussian_noise = 0.005
    #                classifier, history = model(train_data, train_labels,dev_data, dev_labels, vocab, labels, args)
   
   
if __name__ == '__main__':
    main()

