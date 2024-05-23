# TRGOA
TRGOA: Topological-aware Residue-Gene Ontology Attention Network for Protein Function Prediction

## Setup Environment
torch                         2.2.1
torch_geometric               2.4.0
TRGOA is tested to work under Python 3.6.

## Model training
you can run train.ipynb

## Model test
you can run test.ipynb

# Data Format
For each sequence in PDB-cdhit/PDBmmseq dataset, a serialized dictionary stores the processed features used in TRGOA. Details can be found below  
1. ```data.seq: One-hot encoded primary sequence```  
2. ```data.pssm: Sequence profile constructed from MSA```
3. ```data.x: Residue level sequence embedding generated from ESM-1b```
4. ```data.edge_index: Contact map index```
5. ```data.seq_embed: Sequence level embedding generated from ESMA-1b```
6. ```data.label: GO term annotation```
7. ```data.chain_id: Sequence identifier```

## Citing