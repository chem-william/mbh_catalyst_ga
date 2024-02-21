"""
Written by Jan H. Jensen 2018
"""
import random
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

rdBase.DisableLog("rdApp.error")


class Crossover:
    def __init__(
        self, average_size: float, size_stdev: float, molecule_filter: list[Chem.Mol]
    ) -> None:
        self.average_size = average_size
        self.size_stdev = size_stdev
        self.molecule_filter = molecule_filter

    def cut(self, mol: Chem.Mol) -> Optional[Tuple[Chem.Mol]]:
        """
        Attempts to cut a molecule at a single bond that is not part of a ring using a specific SMARTS pattern.

        The SMARTS pattern "[*]-;!@[*]" is used to identify single bonds between any two atoms that are not in a ring.
        If such a bond is found, the molecule is split at this location, and the resulting fragments are returned
        as molecules with dummy atoms added where the cuts were made.

        Parameters:
        - mol (Chem.Mol): The input molecule to be cut.

        Returns:
        - Optional[Tuple[Chem.Mol]]: The resulting molecule fragments as Chem.Mol objects if the cut was successful;
        None if no suitable bond was found or an error occurred during fragmentation.

        Note:
        - This function uses randomness to select among multiple possible cut sites if more than one is found.
        """
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
            return None
        bis = random.choice(
            mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]"))
        )
        bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
            return fragments
        except:
            return None

    def cut_ring(self, mol):
        for i in range(10):
            if np.random.random() < 0.5:
                if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")):
                    return None
                bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")))
                bis = (
                    (bis[0], bis[1]),
                    (bis[2], bis[3]),
                )
            else:
                if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
                    return None
                bis = np.random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")))
                bis = (
                    (bis[0], bis[1]),
                    (bis[1], bis[2]),
                )

            bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

            fragments_mol = Chem.FragmentOnBonds(
                mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)]
            )

            try:
                fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
            except:
                return None

            if len(fragments) == 2:
                return fragments

        return None

    def ring_OK(self, mol):
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
            return True

        ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))

        cycle_list = mol.GetRingInfo().AtomRings()
        max_cycle_length = max([len(j) for j in cycle_list])
        macro_cycle = max_cycle_length > 6

        double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts("[r3,r4]=[r3,r4]"))

        return not ring_allene and not macro_cycle and not double_bond_in_small_ring

    def mol_OK(self, mol) -> bool:
        if not self.size_stdev or not self.average_size:
            print("size parameters are not defined")
        try:
            Chem.SanitizeMol(mol)
            test_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            if test_mol == None:
                return None
            if not self.mol_is_sane(mol):
                return False
            target_size = self.size_stdev * np.random.randn() + self.average_size  # parameters set in GA_mol
            target_nrb = 2 * np.random.randn() + 5
            if target_nrb < 5:
                target_nrb = 5
            if (
                mol.GetNumAtoms() > 5
                and mol.GetNumAtoms() < target_size
                and CalcNumRotatableBonds(mol) < target_nrb
            ):
                return True
            else:
                return False
        except:
            return False

    def mol_is_sane(self, mol) -> bool:
        if self.molecule_filter is None:
            return True

        for pattern in self.molecule_filter:
            if mol.HasSubstructMatch(pattern):
                return False

        return True

    def crossover_ring(self, parent_A: Chem.Mol, parent_B: Chem.Mol):
        ring_smarts = Chem.MolFromSmarts("[R]")
        if not parent_A.HasSubstructMatch(ring_smarts) and not parent_B.HasSubstructMatch(ring_smarts):
            return None

        rxn_smarts1 = [
            "[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]",
            "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]",
        ]
        rxn_smarts2 = [
            "([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]",
            "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]",
        ]
        for i in range(10):
            fragments_A = self.cut_ring(parent_A)
            fragments_B = self.cut_ring(parent_B)
            # print [Chem.MolToSmiles(x) for x in list(fragments_A)+list(fragments_B)]
            if fragments_A == None or fragments_B == None:
                return None

            new_mol_trial = []
            for rs in rxn_smarts1:
                rxn1 = AllChem.ReactionFromSmarts(rs)
                new_mol_trial = []
                for fa in fragments_A:
                    for fb in fragments_B:
                        new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

            new_mols = []
            for rs in rxn_smarts2:
                rxn2 = AllChem.ReactionFromSmarts(rs)
                for m in new_mol_trial:
                    m = m[0]
                    if self.mol_OK(m):
                        new_mols += list(rxn2.RunReactants((m,)))

            new_mols2 = []
            for m in new_mols:
                m = m[0]
                if self.mol_OK(m) and self.ring_OK(m):
                    new_mols2.append(m)

            if len(new_mols2) > 0:
                return random.choice(new_mols2)

        return None


    def crossover_non_ring(self, parent_A: Chem.Mol, parent_B: Chem.Mol):
        for _ in range(10):
            fragments_A = self.cut(parent_A)
            fragments_B = self.cut(parent_B)
            if fragments_A == None or fragments_B == None:
                return None

            rxn = AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")
            new_mol_trial = []
            for fa in fragments_A:
                for fb in fragments_B:
                    new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

            new_mols = []
            for mol in new_mol_trial:
                mol = mol[0]
                if self.mol_OK(mol):
                    new_mols.append(mol)

            if len(new_mols) > 0:
                return random.choice(new_mols)

        return None

    def crossover(
        self, parent_A: Chem.Mol, parent_B: Chem.Mol
    ):
        parent_smiles = [Chem.MolToSmiles(parent_A), Chem.MolToSmiles(parent_B)]
        try:
            Chem.Kekulize(parent_A, clearAromaticFlags=True)
            Chem.Kekulize(parent_B, clearAromaticFlags=True)
        except:
            pass
        for _ in range(10):
            if np.random.random() <= 0.5:
                new_mol = self.crossover_non_ring(parent_A, parent_B)
            else:
                new_mol = self.crossover_ring(parent_A, parent_B)

            if new_mol != None:
                new_smiles = Chem.MolToSmiles(new_mol)
            if new_mol != None and new_smiles not in parent_smiles:
                return new_mol
        return None


if __name__ == "__main__":
    smiles1 = "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1"
    smiles2 = "C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1"

    smiles1 = "Cc1ccc(S(=O)(=O)N2C(N)=C(C#N)C(c3ccc(Cl)cc3)C2C(=O)c2ccccc2)cc1"
    smiles2 = "CC(C#N)CNC(=O)c1cccc(Oc2cccc(C(F)(F)F)c2)c1"

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    child = Crossover.crossover(mol1, mol2)
    mutation_rate = 1.0
    # mutated_child = mutate(child,mutation_rate)

    for i in range(100):
        child = Crossover.crossover(mol1, mol2)
