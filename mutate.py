"""
Written by Jan H. Jensen 2018
"""

import random
from typing import List

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from crossover import Crossover

rdBase.DisableLog("rdApp.error")

RxnSMARTS = str


def delete_atom() -> RxnSMARTS:
    choices = [
        "[*:1]~[D1]>>[*:1]",
        "[*:1]~[D2]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]",
        "[*:1]~[D4](~[*;!H0:2])(~[*;!H0:3])~[*:4]>>[*:1]-[*:2]-[*:3]-[*:4]",
        "[*:1]~[D4](~[*;!H0;!H1:2])(~[*:3])~[*:4]>>[*:1]-[*:2](-[*:3])-[*:4]",
    ]
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]

    return np.random.choice(choices, p=p)


def append_atom() -> RxnSMARTS:
    choices = [
        ["single", ["C", "N", "O", "F", "S", "Cl", "Br"], 7 * [1.0 / 7.0]],
        ["double", ["C", "N", "O"], 3 * [1.0 / 3.0]],
        ["triple", ["C", "N"], 2 * [1.0 / 2.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == "single":
        rxn_smarts = "[*;!H0:1]>>[*:1]X".replace("X", "-" + new_atom)
    if BO == "double":
        rxn_smarts = "[*;!H0;!H1:1]>>[*:1]X".replace("X", "=" + new_atom)
    if BO == "triple":
        rxn_smarts = "[*;H3:1]>>[*:1]X".replace("X", "#" + new_atom)

    return rxn_smarts


def insert_atom() -> RxnSMARTS:
    choices = [
        ["single", ["C", "N", "O", "S"], 4 * [1.0 / 4.0]],
        ["double", ["C", "N"], 2 * [1.0 / 2.0]],
        ["triple", ["C"], [1.0]],
    ]
    p_BO = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=p_BO)

    BO, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if BO == "single":
        rxn_smarts = "[*:1]~[*:2]>>[*:1]X[*:2]".replace("X", new_atom)
    if BO == "double":
        rxn_smarts = "[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]".replace("X", new_atom)
    if BO == "triple":
        rxn_smarts = "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]".replace("X", new_atom)

    return rxn_smarts


def change_bond_order() -> RxnSMARTS:
    choices = [
        "[*:1]!-[*:2]>>[*:1]-[*:2]",
        "[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]",
        "[*:1]#[*:2]>>[*:1]=[*:2]",
        "[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]",
    ]
    p = [0.45, 0.45, 0.05, 0.05]

    return np.random.choice(choices, p=p)


def delete_cyclic_bond() -> RxnSMARTS:
    return "[*:1]@[*:2]>>([*:1].[*:2])"


def add_ring() -> RxnSMARTS:
    choices = [
        "[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1",
        "[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1",
        "[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1",
    ]
    p = [0.05, 0.05, 0.45, 0.45]

    return np.random.choice(choices, p=p)


def change_atom(mol: Chem.Mol) -> RxnSMARTS:
    choices = ["#6", "#7", "#8", "#9", "#16", "#17", "#35"]
    p = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    X = np.random.choice(choices, p=p)
    while not mol.HasSubstructMatch(Chem.MolFromSmarts("[" + X + "]")):
        X = np.random.choice(choices, p=p)
    Y = np.random.choice(choices, p=p)
    while Y == X:
        Y = np.random.choice(choices, p=p)

    return "[X:1]>>[Y:1]".replace("X", X).replace("Y", Y)


def mutate(mol: Chem.Mol, co: Crossover):
    Chem.Kekulize(mol, clearAromaticFlags=True)
    p = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for _ in range(10):
        rxn_smarts_list = [
            insert_atom(),
            change_bond_order(),
            delete_cyclic_bond(),
            add_ring(),
            delete_atom(),
            change_atom(mol),
            append_atom(),
        ]
        rxn_smarts = np.random.choice(rxn_smarts_list, p=p)

        rxn = AllChem.ReactionFromSmarts(rxn_smarts)

        new_mol_trial = rxn.RunReactants((mol,))

        new_mols = []
        for m in new_mol_trial:
            m = m[0]
            if co.mol_OK(m) and co.ring_OK(m):
                new_mols.append(m)

        if len(new_mols) > 0:
            return np.random.choice(new_mols)

    return None


if __name__ == "__main__":
    pass
