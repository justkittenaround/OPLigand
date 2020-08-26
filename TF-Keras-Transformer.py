# https://towardsdatascience.com/machine-translation-with-the-seq2seq-model-different-approaches-f078081aaa37

import csv
import random
import sys, os
import numpy as np
import matplotlib.pyplot as plt




DATA_PATH = './dataset_new.csv'






#read in the receptor fasta as keys and ligand smile as values (list for multiple ligands)
metadata = {}
with open(DATA_PATH) as f:
    r = csv.DictReader(f, delimiter=',')
    for row in r:
        metadata.update({row['Target FASTA']: row['Ligand SMILE(s)']})

#get rid of any pairs without fasta, smile sequences
dataset = {k: v for k,v in metadata.items() if v != '[]' and k != '[]'}
print('Number of Receptor Ligand Pairs is', len(dataset), '!')

#shuffle data
l = list(dataset.items())
random.shuffle(l)
dataset = dict(l)

#seperate inputs (receptors) and ligands (targets), add start/end markers
receptors = []
ligands = []
r_lengths = []
l_lengths = []
for k,v in dataset.items():
    if len(k.split(', ')) >1 :
        for r in k.split(', '):
            r_lengths.append(len(r))
            if len(v.split(', ')) > 1:
                 for l in v.split(', '):
                    if l[0] != '[':
                        l = l.replace("'", '<', 1)
                    elif l[0] == "[":
                        l = l.replace("['", '<', 1)
                    if l[-1] != ']':
                        l = l.replace("'", '>', -1)
                    elif l[-1] == ']':
                        l = l.replace("']", '>', -1)
                    receptors.append(r[2:-2])
                    ligands.append(l)
                    l_lengths.append(len(l))
            else:
                receptors.append(r[2:-2])
                ligands.append(v[2:-2:])
                l_lengths.append(len(l))
    else:
        r_lengths.append(len(k))
        if len(v.split(', ')) > 1:
             for l in v.split(', '):
                if l[0] != '[':
                    l = l.replace("'", '<', 1)
                elif l[0] == "[":
                    l = l.replace("['", '<', 1)
                if l[-1] != ']':
                    l = l.replace("'", '>', -1)
                elif l[-1] == ']':
                    l = l.replace("']", '>', -1)
                receptors.append(k[2:-2])
                ligands.append(l)
                l_lengths.append(len(l))
        else:
            receptors.append(k[2:-2])
            ligands.append(v[2:-2:])
            l_lengths.append(len(l))

if len(receptors) == len(ligands):
    print('Number of Receptor:Ligand Pairs: ', len(receptors), '!')
else:
    print('Bad Data... Receptor:Ligand Pairs do NOT match!!!')

max_encoder = max(r_lengths)
max_decoder = max(l_lengths)

#create tokens
receptor_tokens = set()
ligand_tokens = set()
for receptor in receptors:
    for token in receptor:
        if token not in receptor_tokens:
            receptor_tokens.add(token)

for ligand in ligands:
    for token in ligand:
        if token not in ligand_tokens:
            ligand_tokens.add(token)


input_tokens = sorted(list(receptor_tokens))
target_tokens = sorted(list(ligand_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)


#create input and target features dictionary for encoding and decoding chara/numbers
input_features_dict = dict([(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict([(token, i) for i, token in enumerate(target_tokens)])
reverse_input_features_dict = dict((i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict((i, token) for token, i in target_features_dict.items())


#traiing data setup for one-hot encoders input, decoder input and decoder output
encoder_input_data = np.zeros((len(receptors), max_encoder, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(receptors), max_decoder, num_decoder_tokens),dtype='float32')
decoder_target_data = np.zeros((len(receptors), max_decoder, num_decoder_tokens), dtype='float32')

for line, (receptor, ligand) in enumerate(zip(receptors, ligands)):
    for timestep, token in enumerate(receptor):
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.
    for timestep, token in enumerate(ligand):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        if timestep > 0:
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.

print('Data shape is (line, timestep, # tokens).')
print('Encoder Input:', encoder_input_data.shape)
print('Decoder Input:',decoder_input_data.shape)
print('Decoder Target:',decoder_target_data.shape)



#TF-Keras Model
from tensorflow import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

#Dimensionality
dimensionality = 256

#The batch size and number of epochs
batch_size = 256
epochs = 100

#Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]#Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#Model
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

#Compiling
training_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], sample_weight_mode='temporal')

#Training
training_model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size = batch_size, epochs = epochs, validation_split = 0.2)



















#
