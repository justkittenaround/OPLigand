#torch text
#https://www.youtube.com/watch?v=KRgq4VnCr7I


import csv
import random
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_PATH = './opl_data/dataset_new.csv'


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
                        l = l.replace("'", '', 1)
                    elif l[0] == "[":
                        l = l.replace("['", '', 1)
                    if l[-1] != ']':
                        l = l.replace("'", '', -1)
                    elif l[-1] == ']':
                        l = l.replace("']", '', -1)
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
                    l = l.replace("'", '', 1)
                elif l[0] == "[":
                    l = l.replace("['", '', 1)
                if l[-1] != ']':
                    l = l.replace("'", '', -1)
                elif l[-1] == ']':
                    l = l.replace("']", '', -1)
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

max_r = max(r_lengths)
max_l = max(l_lengths)

d = {'Receptors': receptors, 'Ligands': ligands}

df = pd.DataFrame(d, columns=['Receptors', 'Ligands'])

train, test = train_test_split(df, test_size=0.3)

train.to_json('train.json', orient='records', lines=True)
test.to_json('test.json', orient='records', lines=True)
























#
