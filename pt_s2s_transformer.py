#seq2seq transformer (machine translation)
# https://github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, TabularDataset, BucketIterator

from pt_s2s_utils import translate_sentence, bleu, save_checkpoint, load_checkpoint


import warnings
warnings.filterwarnings("ignore")

import wandb
wandb.init(project='OPL_transformer')
wab = wandb.config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device type: {device} !')

MULTI_GPU = False
LOAD_MODEL = False
SAVE_MODEL = False
MODEL_PATH = 'my_checkpoint.pth.tar'
DATA_FOLDER = 'opl_data'
BS = 8
N_EPOCHS = 100
LR = .0003

embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
p = 0.2
max_len = 1001
forward_expansion = 4


if MULTI_GPU == True:
    BS = BS*torch.cuda.device_count()


class DataParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)
        self.block3 = nn.Linear(20, 20)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x



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
    def __init__(self, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_encoderr_layers, num_decoder_layers, forward_expansion, p, max_len, device):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)
        self.device = device
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, p)
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(p)
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



model = Transformer(embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, p, max_len, device).to(device)

if MULTI_GPU == True:
    if torch.cuda.device_count() > 1:
                  print("Let's use", torch.cuda.device_count(), "GPUs!")
                  model = nn.DataParallel(model)


criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

#optimizer = optim.Adam(model.parameters(), lr=LR)
optimizer = optim.SGD(model.parameters(), lr=LR)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

if LOAD_MODEL:
    load_checkpoint(torch.load(MODEL_PATH), model, optimizer)



model.train()

for epoch in range(N_EPOCHS):
    # print(f'Epoch [{epoch} / {N_EPOCHS}]')
    losses = []
    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        #input shape: (maxlen(in-seqs), BS)
        target = batch.trg.to(device)
        #target shape: (maxlen(targ-seqs), BS)
        optimizer.zero_grad()
        output = model(inp_data, target[:-1])
        # output shape: (maxlen(trg-seqs), BS, len(targ_vocab))
        output = output.reshape(-1, output.shape[2])
        #output shape: (len(seq)-1*BS, len(targ_vocab))
        target = target[1:].reshape(-1)
        loss = criterion(output, target)
        losses.append(loss.detach().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=.5)
        optimizer.step()
        wandb.log({'Training loss': loss}, step=epoch)
        # print(output)
        # print(target)
        # print(loss)
    mean_loss = sum(losses)/len(losses)
    scheduler.step()
    print(f'Loss[{mean_loss}]')
    if SAVE_MODEL and epoch == N_EPOCHS-1:
        checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
        save_checkpoint(checkpoint)




model.eval()

sentence = 'MNNFILLEEQLIKKSQQKRRTSPSNFKVRFFVLTKASLAYFEDRHGKKRTLKGSIELSRIKCVEIVKSDISIPCHYKYPFQVVHDNYLLYVFAPDRESRQRWVLALKEETRNNNSLVPKYHPNFWMDGKWRCCSQLEKLATGCAQYDPTKNASKKPLPPTPEDNRRPLWEPEETVVIALYDYQTNDPQELALRRNEEYCLLDSSEIHWWRVQDRNGHEGYVPSSYLVEKSPNNLETYEWYNKSISRDKAEKLLLDTGKEGAFMVRDSRTAGTYTVSVFTKAVVSENNPCIKHYHIKETNDNPKRYYVAEKYVFDSIPLLINYHQHNGGGLVTRLRYPVCFGRQKAPVTAGLRYGKWVIDPSELTFVQEIGSGQFGLVHLGYWLNKDKVAIKTIREGAMSEEDFIEEAEVMMKLSHPKLVQLYGVCLEQAPICLVFEFMEHGCLSDYLRTQRGLFAAETLLGMCLDVCEGMAYLEEACVIHRDLAARNCLVGENQVIKVSDFGMTRFVLDDQYTSSTGTKFPVKWASPEVFSFSRYSSKSDVWSFGVLMWEVFSEGKIPYENRSNSEVVEDISTGFRLYKPRLASTHVYQIMNHCWKERPEDRPAFSRLLRQLAEIAESGL'
actual = 'COc1cc(C)c(cc1C(=O)N1CCN(CC1)C(=O)C)Sc1cnc(s1)NC(=O)c1ccc(cc1)CNC(C(C)'

translated = translate_sentence(model, sentence, receptors, ligands, device, max_length = 1000)

translated = [chr(i+32) for i in translated]
translated = ''.join(translated)
print(f'Ligand prediction {translated} \n')
print(f'Actual ligand {actual}')



score = bleu(test_data, model, receptors, ligands, device)
print(f"Bleu score {score*100:.2f}")














        #
