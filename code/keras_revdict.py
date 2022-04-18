import argparse
import itertools
import json
import logging
import pathlib
import sys

logger = logging.getLogger(pathlib.Path(__file__).name)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
logger.addHandler(handler)

import tqdm

import data
import models

# additional imports for keras integreation
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


def get_parser(
    parser=argparse.ArgumentParser(
        description="Run a reverse dictionary baseline.\nThe task consists in reconstructing an embedding from the glosses listed in the datasets"
    ),
):
    parser.add_argument(
        "--do_train", action="store_true", help="whether to train a model from scratch"
    )
    parser.add_argument(
        "--do_pred", action="store_true", help="whether to produce predictions"
    )
    parser.add_argument(
        "--train_file", type=pathlib.Path, help="path to the train file"
    )
    parser.add_argument("--dev_file", type=pathlib.Path, help="path to the dev file")
    parser.add_argument("--test_file", type=pathlib.Path, help="path to the test file")
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="path to the train file",
    )
    parser.add_argument(
        "--target_arch",
        type=str,
        default="sgns",
        choices=("sgns", "char", "electra"),
        help="embedding architecture to use as target",
    )
    parser.add_argument(
        "--summary_logdir",
        type=pathlib.Path,
        default=pathlib.Path("logs") / f"revdict-baseline",
        help="write logs for future analysis",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default=pathlib.Path("models") / f"revdict-baseline",
        help="where to save model & vocab",
    )
    parser.add_argument(
        "--pred_file",
        type=pathlib.Path,
        default=pathlib.Path("revdict-baseline-preds.json"),
        help="where to save predictions",
    )
    return parser

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

def train(args):
    assert args.train_file is not None, "Missing dataset for training"
    # 1. get data, vocabulary, summary writer
    logger.debug("Preloading data")
    ## make datasets
    train_dataset = data.JSONDataset(args.train_file)
    if args.dev_file:
        dev_dataset = data.JSONDataset(args.dev_file, vocab=train_dataset.vocab)
    ## assert they correspond to the task
    assert train_dataset.has_gloss, "Training dataset contains no gloss."
    if args.target_arch == "electra":
        assert train_dataset.has_electra, "Training datatset contains no vector."
    else:
        assert train_dataset.has_vecs, "Training datatset contains no vector."
    if args.dev_file:
        assert dev_dataset.has_gloss, "Development dataset contains no gloss."
        if args.target_arch == "electra":
            assert dev_dataset.has_electra, "Development dataset contains no vector."
        else:
            assert dev_dataset.has_vecs, "Development dataset contains no vector."
    ## make dataloader
    train_dataloader = data.get_dataloader(train_dataset, batch_size=512)
    dev_dataloader = data.get_dataloader(dev_dataset, shuffle=False, batch_size=1024)
    ## make summary writer
    summary_writer = SummaryWriter(args.summary_logdir)
    train_step = itertools.count()  # to keep track of the training steps for logging

    # 2. construct model
    ## Hyperparams
    #logger.debug("Setting up training environment")
    #model = models.RevdictModel(dev_dataset.vocab).to(args.device)
    #model.train()
    embedding_size = 300
    hidden_size = 256
    dropout = 0.3
    model = keras.Sequential(
        [
            layers.Embedding(input_dim=len(dev_dataset.vocab), output_dim=embedding_size, weights=[load_pretrained_embeddings('../data/glove.6B/glove.6B.300d.txt', dev_dataset.vocab)], trainable=True, name="GloVe_Embedding"),
            layers.LSTM(hidden_size, return_sequences=True, name="LSTM_1"),
            layers.Dropout(dropout),
            layers.LSTM(hidden_size, return_sequences=False, name="LSTM_2"),
            layers.Dropout(dropout),
            layers.Dense(256,name="Dense")
        ]
    )

    
    # We are going to try a decaying learning rate
    learning_rate = 0.01
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate= learning_rate,decay_steps=1000, decay_rate=0.8)
    
    # Loss is cosine similarity (want to approach -1)
    classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.keras.losses.CosineSimilarity(axis=-1,name='cosine_similarity'), metrics=[metrics.mean_squared_error, 'accuracy'])
    
    # We will store the evaluation metrics for each epoch in this file
    outputFile = open("evaluation.out","w+")
    outputFile.write("Evaluation Information: \n")
    outputFile.write("Epoch\tLoss\tMSE\tAcc\tSet \n")
    outputFile.write("\n")

    epochs = 1
    for i in range(epochs):
        print('Epoch',i+1,'/',args.epochs)
        # Training
        history = History()
        history = model.fit(batch_generator(train_data, train_labels, vocab, labels, batch_size=args.batch_size),
                                 epochs=1, steps_per_epoch=len(train_data)/args.batch_size)
        # Evaluation
        loss, mse, acc = model.evaluate(batch_generator(dev_data, dev_labels, vocab, labels),
                                                  steps=len(dev_data))

        print('Dev Loss:', loss, 'Dev mse:', mse, 'Dev Acc:', acc)

        toWrite1 = str(i) + "\t" + str(history.history['loss']) + "\t" + str(history.history['mean_squared_error']) + "\t" + str(history.history['accuracy']) + "\t" + 'Train' + "\n"
        toWrite2 = str(i) + "\t" + str(loss) + "\t" + str(mse) + "\t" + str(acc) + "\t" + 'Dev/Test' + "\n"
        outputFile.write(toWrite1)
        outputFile.write(toWrite2)

    # Output summary of model layers
    model.summary()

    # 3. declare optimizer & criterion
    ## Hyperparams
    '''
    EPOCHS, LEARNING_RATE, BETA1, BETA2, WEIGHT_DECAY = 1, 1.0e-4, 0.9, 0.999, 1.0e-6 #10
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.MSELoss()

    vec_tensor_key = f"{args.target_arch}_tensor"

    # 4. train model
    for epoch in tqdm.trange(EPOCHS, desc="Epochs"):
        ## train loop
        pbar = tqdm.tqdm(
            desc=f"Train {epoch}", total=len(train_dataset), disable=None, leave=False
        )
        for batch in train_dataloader:
            optimizer.zero_grad()
            gls = batch["gloss_tensor"].to(args.device)
            vec = batch[vec_tensor_key].to(args.device)
            pred = model(gls)
            loss = criterion(pred, vec)
            loss.backward()
            # keep track of the train loss for this step
            next_step = next(train_step)
            summary_writer.add_scalar(
                "revdict-train/cos",
                F.cosine_similarity(pred, vec).mean().item(),
                next_step,
            )
            summary_writer.add_scalar("revdict-train/mse", loss.item(), next_step)
            optimizer.step()
            pbar.update(vec.size(0))
        pbar.close()
        ## eval loop
        if args.dev_file:
            model.eval()
            with torch.no_grad():
                sum_dev_loss, sum_cosine = 0.0, 0.0
                pbar = tqdm.tqdm(
                    desc=f"Eval {epoch}",
                    total=len(dev_dataset),
                    disable=None,
                    leave=False,
                )
                for batch in dev_dataloader:
                    gls = batch["gloss_tensor"].to(args.device)
                    vec = batch[vec_tensor_key].to(args.device)
                    pred = model(gls)
                    sum_dev_loss += (
                        F.mse_loss(pred, vec, reduction="none").mean(1).sum().item()
                    )
                    sum_cosine += F.cosine_similarity(pred, vec).sum().item()
                    pbar.update(vec.size(0))
                # keep track of the average loss on dev set for this epoch
                summary_writer.add_scalar(
                    "revdict-dev/cos", sum_cosine / len(dev_dataset), epoch
                )
                summary_writer.add_scalar(
                    "revdict-dev/mse", sum_dev_loss / len(dev_dataset), epoch
                )
                pbar.close()
            model.train()

    # 5. save result
    model.save(args.save_dir / "model.pt")
    train_dataset.save(args.save_dir / "train_dataset.pt")
    dev_dataset.save(args.save_dir / "dev_dataset.pt")
    '''

def pred(args):
    assert args.test_file is not None, "Missing dataset for test"
    # 1. retrieve vocab, dataset, model
    model = models.DefmodModel.load(args.save_dir / "model.pt")
    train_vocab = data.JSONDataset.load(args.save_dir / "train_dataset.pt").vocab
    test_dataset = data.JSONDataset(
        args.test_file, vocab=train_vocab, freeze_vocab=True, maxlen=model.maxlen
    )
    test_dataloader = data.get_dataloader(test_dataset, shuffle=False, batch_size=1024)
    model.eval()
    vec_tensor_key = f"{args.target_arch}_tensor"
    assert test_dataset.has_gloss, "File is not usable for the task"
    # 2. make predictions
    predictions = []
    with torch.no_grad():
        pbar = tqdm.tqdm(desc="Pred.", total=len(test_dataset))
        for batch in test_dataloader:
            vecs = model(batch["gloss_tensor"].to(args.device)).cpu()
            for id, vec in zip(batch["id"], vecs.unbind()):
                predictions.append(
                    {"id": id, args.target_arch: vec.view(-1).cpu().tolist()}
                )
            pbar.update(vecs.size(0))
        pbar.close()
    with open(args.pred_file, "w") as ostr:
        json.dump(predictions, ostr)


def main(args):
    if args.do_train:
        logger.debug("Performing revdict training")
        train(args)
    if args.do_pred:
        logger.debug("Performing revdict prediction")
        pred(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
