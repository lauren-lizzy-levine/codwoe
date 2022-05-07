import argparse
import json
import os
from collections import Counter

import tensorflow as tf
from tensorflow import keras
import numpy as np
from preproc import stem, lemmatize, lower, rem_stop, rem_punc

# default args
args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-l','--load', default='keras_model', help='Model Folder')
args.add_argument('-t','--test', help='Test File')
args.add_argument('-tr','--train', help='Train File')
args.add_argument('-et','--embedding-type', default='sgns', type=str, help='Embedding type used: char, sgns, or electra')
args.add_argument('-o','--outfile', default='revdict_preds.json', type=str, help='Output file for predictions')
args.add_argument('-pp','--preprocessing', type=str, help='Preprocess the gloss', default='None')

args = args.parse_args()

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
    ids = []
    with open(data_file, 'r', encoding='utf8') as f:
        dataset = json.load(f)
        for line in dataset:
            #print(line)
            gloss = line['gloss'].split(' ')
            adjusted_gloss = [START] + gloss + [END]
            data.append(transform_text_sequence(adjusted_gloss))
            ids.append(line['id'])
            # This is the merge lists approach for labels where there are 768 output dimensions
            # Assumption: Each gloss is unique
            #labels.append(line['char'] + line['sgns'] + line['electra'])
            
            # This is the only use one embedding and go from there
            # There are 256 dimensions in each of the three embeddings
            #labels.append(line[args.embedding_type])
            
            for tok in adjusted_gloss:
                vocab[tok]+=1
           
    vocab = [w for w in sorted(vocab, key=lambda x:vocab[x], reverse=True)]
    if max_vocab_size:
        vocab = vocab[:max_vocab_size-2]
    vocab = [UNK, PAD] + vocab
    return {k:v for v,k in enumerate(vocab)}, data, ids

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

def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]

def batch_generator(data, vocab, batch_size=1):
    '''
    This function takes the input data and labels and outputs lists (batches) with length
    batch_size and padded to be the same length.
    '''
    while True:
        batch_x = []
        for doc in data:
            batch_x.append(vectorize_sequence(doc, vocab))
            if len(batch_x) >= batch_size:
                # Pad Sequences in batch to same length
                batch_x = pad_sequences(batch_x, vocab[PAD])
                yield np.array(batch_x)
                batch_x = []

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

def main():
    vocab, test_data, ids = get_vocabulary_and_data(args.test)
    vocab_train, train_data, train_ids = get_vocabulary_and_data(args.train)
    # doesn't cover padding yet
    input_data = [vectorize_sequence(seq, vocab_train) for seq in test_data]
    model = keras.models.load_model(args.load)
    #input_batch = batch_generator(test_data[:1], vocab_train, batch_size=1)
    #predictions = model.predict(input_batch)
    entries = []
    for identifier, data in zip(ids, input_data):
        print(identifier)
        entries.append(
            {"id": identifier, args.embedding_type: model.predict([data]).tolist()[0]}
        )
    with open(args.outfile, "w") as ostr:
        json.dump(entries, ostr)


if __name__ == '__main__':
    main()
