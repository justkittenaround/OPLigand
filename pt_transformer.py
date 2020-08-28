#torch text
#https://www.youtube.com/watch?v=DaHAzCaXWYQ

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator

from pt_transformer_utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device type: {device} !')

DATA_FOLDER = 'opl_data'
BS = 8
N_EPOCHS = 1
LR = 0.001

encoder_embedding_size = 100
decoder_embedding_size = 100
hidden_size = 1024
n_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5



#how to tokenize the text
def tokenize_r(text):
    input_tokens = [chr(i+65) for i in range(26)]
    input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
    return [input_features_dict[token] for token in text]


def tokenize_l(text):
    target_tokens = [chr(i+32) for i in range(256)]
    target_features_dict = dict((token, i) for i, token in enumerate(target_tokens))
    return [target_features_dict[token] for token in text]


#how data should be stored
receptors = Field(sequential=True, use_vocab=True, tokenize=tokenize_r, lower=False)
ligands = Field(sequential=True, use_vocab=True, tokenize=tokenize_l, lower=False)

#what data to pull from file
fields = {'Receptors': ('src', receptors), 'Ligands':('trg', ligands)}

#make data
train_data, test_data = TabularDataset.splits(path=DATA_FOLDER, train='train.json', test='test.json', format='json', fields=fields)

#build the vocab from the tokenizers
receptors.build_vocab(train_data, max_size=10000, min_freq=1)
ligands.build_vocab(train_data, max_size=10000, min_freq=1)

#create iterators for pulling data
train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=BS, device=device)

input_size_encoder = len(receptors.vocab)
input_size_decoder = len(ligands.vocab)
output_size = len(ligands.vocab)



class Encoder(nn.Module):
    def __init__(self,  input_size, embedding_size, hidden_size, n_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size= hidden_size
        self.n_layers = n_layers
        self.drop = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=p)
    def forward(self, x):
            #xshape: (seq_length, BS)
            embedding = self.drop(self.embedding(x))
            #embedding shape: (seq_length, BS, embedding_dim)
            out, (h,c) = self.rnn(embedding)
            return h,c

class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, n_layers, p):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers= n_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, h, c):
        #shape of x: (BS)--> (1, BS)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        #shape: (1, BS, embedding_dim)
        out, (h,c) = self.rnn(embedding, (h,c))
        #shape: (1, BS, hidden_size)
        preds = self.fc(out)
        #shape = (1, BS, len(vocab))
        preds = preds.squeeze(0)
        return preds, (h,c)

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg, teacher_force_ratio=0.5):
        batch_size = src.shape[1]
        target_len = trg.shape[0]
        target_vocab_size = len(ligands.vocab)
        outs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        h,c = self.encoder(src)
        #start token -->decoder
        x = trg[0]
        for t in range(1, target_len):
            output, (h, c) = self.decoder(x, h,c)
            outs[t] = output
            #output shape: (BS, english_vocab_size)
            best = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best
        return outs



encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, n_layers, enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, input_size_decoder, n_layers, dec_dropout).to(device)

model = Seq2seq(encoder_net, decoder_net).to(device)

# ignore the padding in loss calc
pad_idx = ligands.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

optimizer = optim.Adam(model.parameters(), lr=LR)



for epoch in range(N_EPOCHS):
    print(f'Epoch [{epoch} / {N_EPOCHS}]')
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        #input shape: (maxlen(in-seqs), BS)
        target = batch.trg.to(device)
        #target shape: (maxlen(targ-seqs), BS)
        output = model(inp_data, target)
        #output shape: (maxlen(trg-seqs), BS, len(targ_vocab))
        output = output[1:].reshape(-1, output.shape[2]) #skip start pad
        #output shape: (len(seq)-1*BS, len(targ_vocab))
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    print(f'Loss[{loss.sum()}]')





model.eval()

sentence = 'MNNFILLEEQLIKKSQQKRRTSPSNFKVRFFVLTKASLAYFEDRHGKKRTLKGSIELSRIKCVEIVKSDISIPCHYKYPFQVVHDNYLLYVFAPDRESRQRWVLALKEETRNNNSLVPKYHPNFWMDGKWRCCSQLEKLATGCAQYDPTKNASKKPLPPTPEDNRRPLWEPEETVVIALYDYQTNDPQELALRRNEEYCLLDSSEIHWWRVQDRNGHEGYVPSSYLVEKSPNNLETYEWYNKSISRDKAEKLLLDTGKEGAFMVRDSRTAGTYTVSVFTKAVVSENNPCIKHYHIKETNDNPKRYYVAEKYVFDSIPLLINYHQHNGGGLVTRLRYPVCFGRQKAPVTAGLRYGKWVIDPSELTFVQEIGSGQFGLVHLGYWLNKDKVAIKTIREGAMSEEDFIEEAEVMMKLSHPKLVQLYGVCLEQAPICLVFEFMEHGCLSDYLRTQRGLFAAETLLGMCLDVCEGMAYLEEACVIHRDLAARNCLVGENQVIKVSDFGMTRFVLDDQYTSSTGTKFPVKWASPEVFSFSRYSSKSDVWSFGVLMWEVFSEGKIPYENRSNSEVVEDISTGFRLYKPRLASTHVYQIMNHCWKERPEDRPAFSRLLRQLAEIAESGL'
actual = 'COc1cc(C)c(cc1C(=O)N1CCN(CC1)C(=O)C)Sc1cnc(s1)NC(=O)c1ccc(cc1)CNC(C(C)'

translated = translate_sentence(model, sentence, receptors, ligands, device, max_length = 1000)
translated = [chr(i+32) for i in translated]
translated = ''.join(translated)
print(f'Ligand prediction {translated} \n')
print(f'Actual ligand {actual}')



score = bleu(test_data[1:100], model, receptors, ligands, device)
print(f"Bleu score {score*100:.2f}")


















#
