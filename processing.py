import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import tqdm
import torch
from sklearn.utils import shuffle

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
word_dict = defaultdict(lambda: len(word_dict))


def create_atoms(mol):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    atoms_set = set(range(mol.GetNumAtoms()))
    isolate_atoms = atoms_set - set(i_jbond_dict.keys())
    bond = bond_dict['nan']
    for a in isolate_atoms:
        i_jbond_dict[a].append((a, bond))

    return i_jbond_dict


def atom_features(atoms, i_jbond_dict, radius):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]
    
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])

            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)
    return adjacency


def get_fingerprints(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useChirality=True)
    return fp.ToBitString()


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

def precessing(input_path, output_path, radius, ngram):
    data = pd.read_table(input_path + '.txt', sep=' ',header=None)
    print("shuffle dataset...")
    data = shuffle(data)
    data = shuffle(data)
    compounds, adjacencies, fps, proteins, interactions = [], [], [], [], []
    for index in tqdm.tqdm(range(len(data))):
        smiles, sequence, interaction = data.iloc[index, :]
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol)
        i_jbond_dict = create_ijbonddict(mol)
        compounds.append(atom_features(atoms, i_jbond_dict, radius))
        adjacencies.append(create_adjacency(mol))
        fps.append(get_fingerprints(mol))
        proteins.append(split_sequence(sequence, ngram))
        interactions.append(np.array([float(interaction)]))
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'compounds'), compounds)
    np.save(os.path.join(output_path, 'adjacencies'), adjacencies)
    np.save(os.path.join(output_path, 'fingerprint'), fps)
    np.save(os.path.join(output_path, 'proteins'), proteins)
    np.save(os.path.join(output_path, 'interactions'), interactions)

def process(dataset):
    radius, ngram = 2, 3
    input_path = os.path.join('./data', dataset)
    output_path = os.path.join('./datasets',  dataset)
    precessing(input_path, output_path, radius, ngram)

    dump_dictionary(fingerprint_dict, os.path.join('./datasets',  dataset, 'atom_dict'))
    dump_dictionary(word_dict, os.path.join('./datasets',dataset, 'amino_dict'))
