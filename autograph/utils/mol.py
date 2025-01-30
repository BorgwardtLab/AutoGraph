import torch
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
RDLogger.DisableLog('rdApp.*')


atom_types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'B': 4, 'Br': 5, 'Cl': 6, 'I': 7, 'P': 8, 'S': 9, 'Se': 10, 'Si': 11}
bond_types = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

def smiles2graph(smile):
    mol = Chem.MolFromSmiles(smile)
    N = mol.GetNumAtoms()

    type_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(atom_types[atom.GetSymbol()])

    row, col, edge_type = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bond_types[bond.GetBondType()]]

    if len(row) == 0:
        return None

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    x = torch.tensor(type_idx)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_type)
    return data
