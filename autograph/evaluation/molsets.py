###############################################################################
#
# Adapted from https://github.com/cvignac/DiGress
#
###############################################################################
import numpy as np
import torch
import re
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric import utils


allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1, 'B': 3, 'Al': 3, 'Si': 4, 'P': [3, 5],
                 'S': 4, 'Cl': 1, 'As': 3, 'Br': 1, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class BasicMolecularMetrics(object):
    def __init__(self, atom_decoder, train_smiles=None):
        self.atom_decoder = atom_decoder

        # Retrieve dataset smiles only for qm9 currently.
        self.dataset_smiles_list = train_smiles

    def compute_validity(self, generated):
        """ generated: list of couples (positions, atom_types)"""
        valid = []
        num_components = []
        all_smiles = []
        for graph in generated:
            mol = build_molecule(graph, self.atom_decoder)
            smiles = mol2smiles(mol)
            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                    all_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        return valid, len(valid) / len(generated), np.array(num_components), all_smiles

    def compute_uniqueness(self, valid):
        """ valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return 1, 1
        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in generated:
            mol = build_molecule_with_partial_charges(graph, self.atom_decoder)
            smiles = mol2smiles(mol)
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    valid.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
        return valid, len(valid) / len(generated)

    def evaluate(self, generated):
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        valid, validity, num_components, all_smiles = self.compute_validity(generated)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        print(f"Number of connected components of {len(generated)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(generated)
        print(f"Relaxed validity over {len(generated)} molecules: {relaxed_validity * 100 :.2f}%")
        if relaxed_validity > 0:
            unique, uniqueness = self.compute_uniqueness(relaxed_valid)
            print(f"Uniqueness over {len(relaxed_valid)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                _, novelty = self.compute_novelty(unique)
                print(f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique = []

        if self.atom_decoder[0] == 'H':
            stability_dict = check_stability(generated, self.atom_decoder)
        else:
            stability_dict = {'mol_stable': -1, 'atm_stable': -1}
        return ([validity, relaxed_validity, uniqueness, novelty], stability_dict, unique,
                dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu), all_smiles)


def data2graph(data):
    if isinstance(data, Data):
        node_types = data.x
        edge_types = utils.to_torch_coo_tensor(
            data.edge_index,
            data.edge_attr + 1,
            data.num_nodes
        ).to_dense()
        return node_types, edge_types
    return data


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def build_molecule(data, atom_decoder, verbose=False):
    if verbose:
        print("building new molecule")

    atom_types = data.x
    edge_types = data.edge_attr
    all_bonds = [tuple(i) for i in data.edge_index.t().tolist()]

    mol = Chem.RWMol()

    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    visited = set()
    for i in range(len(all_bonds)):
        bond = all_bonds[i]
        if tuple(sorted(bond)) in visited:
            continue
        if bond[0] != bond[1]:
            mol.AddBond(bond[0], bond[1], bond_dict[edge_types[i].item()])
            if verbose:
                print("bond added:", bond[0], bond[1], edge_types[i].item(),
                      bond_dict[edge_types[i].item()] )
        visited.add(tuple(sorted(bond)))
    return mol


def build_molecule_with_partial_charges(data, atom_decoder, verbose=False):
    if verbose:
        print("\nbuilding new molecule")
    atom_types = data.x
    edge_types = data.edge_attr
    all_bonds = [tuple(i) for i in data.edge_index.t().tolist()]

    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)
        if verbose:
            print("Atom added: ", atom.item(), atom_decoder[atom.item()])

    visited = set()
    for i in range(len(all_bonds)):
        bond = all_bonds[i]
        if tuple(sorted(bond)) in visited:
            continue
        visited.add(tuple(sorted(bond)))
        if bond[0] != bond[1]:
            mol.AddBond(bond[0], bond[1], bond_dict[int(edge_types[i])])
            if verbose:
                print("bond added:", bond[0], bond[1], int(edge_types[i]),
                      bond_dict[int(edge_types[i])])
            # add formal charge to atom: e.g. [O+], [N+], [S+]
            # not support [O-], [N-], [S-], [NH+] etc.
            flag, atomid_valence = check_valency(mol)
            if verbose:
                print("flag, valence", flag, atomid_valence)
            if flag:
                continue
            else:
                assert len(atomid_valence) == 2
                idx = atomid_valence[0]
                v = atomid_valence[1]
                an = mol.GetAtomWithIdx(idx).GetAtomicNum()
                if verbose:
                    print("atomic num of atom with a large valence", an)
                if an in (7, 8, 16) and (v - ATOM_VALENCY[an]) == 1:
                    mol.GetAtomWithIdx(idx).SetFormalCharge(1)
    return mol


# Functions from GDSS
def check_valency(mol):
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True, None
    except ValueError as e:
        e = str(e)
        p = e.find('#')
        e_sub = e[p:]
        atomid_valence = list(map(int, re.findall(r'\d+', e_sub)))
        return False, atomid_valence


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and '.' in sm:
        vsm = [(s, len(s)) for s in sm.split('.')]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


def check_stability_one_mol(data, atom_decoder):
    atom_types, edge_types = data2graph(data)

    n_bonds = np.zeros(len(atom_types), dtype='int')

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i])/2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i])/2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)


def check_stability(molecule_list, atom_decoder):
    molecule_stable = 0
    nr_stable_bonds = 0
    n_atoms = 0
    n_molecules = len(molecule_list)

    for i, mol in enumerate(molecule_list):
        validity_results = check_stability_one_mol(mol, atom_decoder)

        molecule_stable += int(validity_results[0])
        nr_stable_bonds += int(validity_results[1])
        n_atoms += int(validity_results[2])

    # Validity
    fraction_mol_stable = molecule_stable / float(n_molecules)
    fraction_atm_stable = nr_stable_bonds / float(n_atoms)
    validity_dict = {'mol_stable': fraction_mol_stable, 'atm_stable': fraction_atm_stable}
    return validity_dict


def compute_molecular_metrics(molecule_list, atom_decoder, train_smiles=None):
    """ molecule_list: (dict) """

    metrics = BasicMolecularMetrics(atom_decoder, train_smiles)
    rdkit_metrics = metrics.evaluate(molecule_list)
    all_smiles = rdkit_metrics[-1]
    nc = rdkit_metrics[-2]
    to_log = {'validity': rdkit_metrics[0][0], 'relaxed_validity': rdkit_metrics[0][1],
           'uniqueness': rdkit_metrics[0][2], 'novelty': rdkit_metrics[0][3],
           'nc_max': nc['nc_max'], 'nc_mu': nc['nc_mu']}
    to_log.update(rdkit_metrics[1])
    to_log['smiles'] = all_smiles

    return to_log
