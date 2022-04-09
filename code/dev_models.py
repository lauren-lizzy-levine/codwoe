import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):
    """From PyTorch"""

    def __init__(self, d_model, dropout=0.1, max_len=4096):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class RevdictModel(nn.Module):
    """A Bi-Directional LSTM architecture for Reverse Dictionary Modeling.""" # turn this into a bi-directional LSTM

    def __init__(
        self, vocab, d_model=256, n_head=4, n_layers=4, dropout=0.3, maxlen=512
    ):
        super(RevdictModel, self).__init__()
        self.d_model = d_model
        self.padding_idx = vocab[data.PAD]
        self.eos_idx = vocab[data.EOS]
        self.maxlen = maxlen

        self.embedding = nn.Embedding(len(vocab), d_model, padding_idx=self.padding_idx)
        self.positional_encoding = PositionalEncoding(
            d_model, dropout=dropout, max_len=maxlen
        )
        #encoder_layer = nn.TransformerEncoderLayer(
        #    d_model=d_model, nhead=n_head, dropout=dropout, dim_feedforward=d_model * 2
        #)
        #self.transformer_encoder = nn.TransformerEncoder(
        #    encoder_layer, num_layers=n_layers
        #)
        
        lstm_units = 50
        hidden_dim = 30
        self.lstm = nn.LSTM(d_model,
                            lstm_units,
                            num_layers=n_layers,
                            bidirectional=True,
                            batch_first=True)
        '''
        num_directions = 2
        self.fc1 = nn.Linear(lstm_units * num_directions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = n_layers
        self.num_directions = num_directions
        self.lstm_units = lstm_units
        '''
        self.dropout = nn.Dropout(p=dropout)
        self.e_proj = nn.Linear(d_model, d_model)
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:  # gain parameters of the layer norm
                nn.init.ones_(param)
    '''
    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)),
            Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)))
        return h, c


    def forward(self, gloss_tensor):
        lengths = []
        for x in gloss_tensor:
            lengths.append(len(x))
        #print(gloss_tensor.shape)
        batch_size = gloss_tensor.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = self.embedding(gloss_tensor)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
        output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0))
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        out = output_unpacked[:, -1, :]
        rel = self.relu(out)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        return preds
    '''

    def forward(self, gloss_tensor):
        src_key_padding_mask = gloss_tensor == self.padding_idx
        embs = self.embedding(gloss_tensor)
        src = self.positional_encoding(embs)
        lstm_output = self.lstm(src)#self.dropout(
            #self.lstm(src)#, src_key_padding_mask=src_key_padding_mask.t())
        #)
        #summed_embs = lstm_output.masked_fill(
        #    src_key_padding_mask.unsqueeze(-1), 0
        #).sum(dim=0)
        #drop = self.dropout(lstm_output)
        return self.e_proj(F.relu(lstm_output)) #summed_embs

    @staticmethod
    def load(file):
        return torch.load(file)

    def save(self, file):
        torch.save(self, file)
