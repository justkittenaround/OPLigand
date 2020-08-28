#seq2seq transformer (machine translation)
# https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator

from pt_s2s_utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device type: {device} !')


LOAD_MODEL = False
SAVE_MODEL = True
MODEL_PATH = 'my_checkpoint.pth.tar'
DATA_FOLDER = 'opl_data'
BS = 8
N_EPOCHS = 1
LR = 3e-4

embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 1000
forward_expansion = 4




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
receptors = Field(sequential=True, use_vocab=True, tokenize=tokenize_r, lower=False, init_token='b', eos_token='j')
ligands = Field(sequential=True, use_vocab=True, tokenize=tokenize_l, lower=False, init_token='*', eos_token='!')

#what data to pull from file
fields = {'Receptors': ('src', receptors), 'Ligands':('trg', ligands)}

#make data
train_data, test_data = TabularDataset.splits(path=DATA_FOLDER, train='train.json', test='test.json', format='json', fields=fields)

#build the vocab from the tokenizers
receptors.build_vocab(train_data, max_size=10000, min_freq=1)
ligands.build_vocab(train_data, max_size=10000, min_freq=1)

#create iterators for pulling data
train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), batch_size=BS, sort_within_batch=True, sort_key=lambda x:len(x.src), device=device)

src_vocab_size = len(receptors.vocab)
trg_vocab_size = len(ligands.vocab)
src_pad_idx = ligands.vocab.stoi['<pad>']




class Transformer(nn.Module):
    def __init__(self, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_encoderr_layers, num_decoder_layers, forward_expansion, dropout, max_len, device):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout)
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
    def make_src_mask(self, src):
        src_mask = src.transpose(0,1) == self.src_pad_idx
        return src_mask.to(self.device)
    def forward(self, src, trg):
        src_seq_len, N = src.shape
        trg_seq_len, N = trg.shape
        src_positions = (torch.arange(0, src_seq_len).unsqueeze(1).expand(src_seq_len, N).to(self.device))
        trg_positions = (torch.arange(0, trg_seq_len).unsqueeze(1).expand(trg_seq_len, N).to(self.device))
        embed_src = self.dropout((self.src_word_embedding(src)+self.src_position_embedding(src_positions)))
        embed_trg = self.dropout((self.trg_word_embedding(trg)+self.trg_position_embedding(trg_positions)))
        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(self.device)
        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)
        out = self.fc_out(out)
        return out



model = Transformer(embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, max_len, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)

if LOAD_MODEL:
    load_checkpoint(torch.load(MODEL_PATH), model, optimizer)



for epoch in range(N_EPOCHS):
    print(f'Epoch [{epoch} / {N_EPOCHS}]')
    model.train()
    losses = []
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        #input shape: (maxlen(in-seqs), BS)
        target = batch.trg.to(device)
        #target shape: (maxlen(targ-seqs), BS)
        output = model(inp_data, target)
        # output shape: (maxlen(trg-seqs), BS, len(targ_vocab))
        output = output[1:].reshape(-1, output.shape[2]) #skip start pad
        #output shape: (len(seq)-1*BS, len(targ_vocab))
        target = target[1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        losses.append(loss.detach().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
    mean_loss = sum(losses)/len(losses)
    scheduler.step(mean_loss)
    print(f'Loss[{mean_loss}]')
    if SAVE_MODEL and epoch == N_EPOCHS-1:
        checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)



#
# model.eval()
#
# sentence = 'MNNFILLEEQLIKKSQQKRRTSPSNFKVRFFVLTKASLAYFEDRHGKKRTLKGSIELSRIKCVEIVKSDISIPCHYKYPFQVVHDNYLLYVFAPDRESRQRWVLALKEETRNNNSLVPKYHPNFWMDGKWRCCSQLEKLATGCAQYDPTKNASKKPLPPTPEDNRRPLWEPEETVVIALYDYQTNDPQELALRRNEEYCLLDSSEIHWWRVQDRNGHEGYVPSSYLVEKSPNNLETYEWYNKSISRDKAEKLLLDTGKEGAFMVRDSRTAGTYTVSVFTKAVVSENNPCIKHYHIKETNDNPKRYYVAEKYVFDSIPLLINYHQHNGGGLVTRLRYPVCFGRQKAPVTAGLRYGKWVIDPSELTFVQEIGSGQFGLVHLGYWLNKDKVAIKTIREGAMSEEDFIEEAEVMMKLSHPKLVQLYGVCLEQAPICLVFEFMEHGCLSDYLRTQRGLFAAETLLGMCLDVCEGMAYLEEACVIHRDLAARNCLVGENQVIKVSDFGMTRFVLDDQYTSSTGTKFPVKWASPEVFSFSRYSSKSDVWSFGVLMWEVFSEGKIPYENRSNSEVVEDISTGFRLYKPRLASTHVYQIMNHCWKERPEDRPAFSRLLRQLAEIAESGL'
# actual = 'COc1cc(C)c(cc1C(=O)N1CCN(CC1)C(=O)C)Sc1cnc(s1)NC(=O)c1ccc(cc1)CNC(C(C)'
#
# translated = translate_sentence(model, sentence, receptors, ligands, device, max_length = 1000)
# translated = [chr(i+32) for i in translated]
# translated = ''.join(translated)
# print(f'Ligand prediction {translated} \n')
# print(f'Actual ligand {actual}')
#


# score = bleu(test_data, model, receptors, ligands, device)
# print(f"Bleu score {score*100:.2f}")














        #
