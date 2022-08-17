# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 09:54:07 2022

@author: DELL
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.metrics import pairwise_distances

compounds = pd.read_excel('data/SigmaMetList.xlsx')
mols = [Chem.MolFromSmiles(s) for s in compounds['SMILES']]

mols = [m for m in mols if m is not None]
fps = [np.array(AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits = 4096)) for m in mols]
fps = np.array(fps)

sim_matrix = 1 - pairwise_distances(fps, metric='jaccard', n_jobs=-1)
sim_matrix = sim_matrix - np.eye(len(fps))

sim_max = np.max(sim_matrix, axis = 1)
