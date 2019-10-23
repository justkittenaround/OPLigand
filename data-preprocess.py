"
!pip install pubchempy
!pip install pubchemprops
!pip install scipy==1.1.0
"



import pubchempy as pcp
from pubchemprops.pubchemprops import *
from urllib.request import Request, urlopen
import numpy as np
import pandas as pd
import os, sys
import time
from scipy.misc import imsave


FILEPATH = 'LigandReceptorDatabase/interactions.csv'
SAVE_PATH = '/home/whale/Desktop/Rachel/OpLigand/DATA/ONE-HOT/receptors'



def load_raw_data(FILEPATH):
    dataset = np.asarray(pd.read_csv(FILEPATH, delimiter=',', header=(0), encoding='utf-8'))
    print(dataset.shape, dataset[0,:])
    return dataset


data = load_raw_data(FILEPATH)
ligand_pubchem_ids = data[:, 13]
receptor_uniprot_ids = data[:,3]


def get_smiles(L_dataset):
    isomeric_smiles = []
    for ligand in L_dataset:
        s = time.time()
        c = pcp.Compound.from_cid(ligand)
        smiles = c.isomeric_smiles
        isomeric_smiles.append(smiles)
    return isomeric_smiles

smiles = get_smiles(ligand_pubchem_ids[0:1])


def get_uniprot_data(ids):
    Protein_data = {} #to store the downloaded sequences from uniprot
    missing = []
    for i in range(len(ids)):
        kw = str(ids[i])
        if kw == 'nan':
            continue
        url1 = 'https://www.uniprot.org/uniprot/'
        url2 = '.fasta'
        query_complete = url1 + kw + url2 #get the url
        request = Request(query_complete) # request connection
        try:
            response = urlopen(request) #open connection after request granted
        except:
            missing.append(kw)
            continue
        data = response.read() #read the data
        data = str(data, 'utf-8') #read the data in the utf-8 format
        data = data.split('\n') #seperate the collected data by '\n'
        data = data[1:-1]
        Protein_data[str(i)] = list(map(lambda x:x.lower(),data))
        x = Protein_data
    return x

proteins_dict = get_uniprot_data(receptor_uniprot_ids)


def convert_hot(data_type):
    if data_type == smiles:
        max_seq_len = len(max(smiles, key=len)) # check max length of element in a list
        hot_ims = []
        for seq in smiles:
            im = np.zeros((256, max_seq_len))
#             h_ala = 256-im.shape(0)
#             div = int(h_ala/im.shape(0))
#             for i in range(div):
#                 if i == 0:
#                     im = np.concatenate((im,im), axis=0)
#                 elif i == 1:
#                     continue
#                 else:
#                     Im = np.concatenate((Im, im). axis=0)
            encoded = np.asarray([ord(char) for char in seq])
            for idx, char in enumerate(encoded):
                im[char, idx] = 1
            hot_ims.append(im)

    return hot_ims


hot_protein_ims = convert_hot(proteins)
hot_smiles_ims = convert_hot(smiles)


def save_images(hot_ims):
    for idx, image in enumerate(hot_ims):
        save_name = SAVE_PATH + '/' + str(ligand_pubchem_ids[idx]) + '.png'
        imsave(save_name, image)
    print('Images saved!')

save_images(hot_protein_ims)
save_images(hot_smiles_ims)





































    #
