from rdkit import Chem
import numpy as np
import pytest

import GA_catalyst
from ..GA_catalyst import GA

from crossover import Crossover
import filters


def is_linkers_ok(molecule: Chem.Mol, ga: GA) -> bool:
    amount_tag = 0
    for atom in molecule.GetAtoms():
        if atom.GetSymbol() == ga.crossover.tagger_atom:
            amount_tag += 1
            num_hs = atom.GetTotalNumHs()

    return amount_tag == 2 and num_hs == 1


@pytest.fixture
def general_crossover_fixture():
    molecule_filters = filters.get_molecule_filters(
        ["MBH"], "./filters/alert_collection.csv"
    )
    co = Crossover(
        average_size=8.0,
        size_stdev=4.0,
        molecule_filter=molecule_filters,
        tagger_atom="Se",
    )
    return co


@pytest.fixture
def general_ga_fixture(general_crossover_fixture):
    population_size = 12
    scoring_function = GA_catalyst.test_scoring
    generations = 32
    mating_pool_size = population_size
    mutation_rate = 0.50
    seed_population = np.load("../../generate_molecules/gdb13/carbon_smiles.npz")[
        "arr_0"
    ]

    prune_population = True
    random_seed = 42
    minimization = True
    selection_method = "rank"
    selection_pressure = 1.5

    ga = GA(
        crossover=general_crossover_fixture,
        population_size=population_size,
        scoring_function=scoring_function,
        generations=generations,
        mating_pool_size=mating_pool_size,
        mutation_rate=mutation_rate,
        prune_population=prune_population,
        random_seed=random_seed,
        minimization=minimization,
        selection_method=selection_method,
        selection_pressure=selection_pressure,
        path=".",
        seed_population=seed_population,
    )
    return ga


@pytest.mark.parametrize(
    "smiles",
    [
        "CCC(C)(CC)C1C=CCC1C",
        "CC(C=C)=CCC=C(C#C)C#C",
        "CC1CC2(C)CCC(=C)CC2=C1",
        "CCC(C)C#CCCCC1CC1",
        "CC1CCC2CCCC(C2)CC1",
        "CC1CCC2C3CC(=C)C2C3C1",
        "CC=C1CC=C2CC=CCC12",
        "C=CC1(CC(=C)CC(=C)C1)C=C",
        "CCCC=CC(C)C(C)C#CC",
        "C=CC1CC#CC=CCCC=C1",
    ],
)
def test_attach_electrodes_correct(smiles, general_ga_fixture):
    mol = Chem.MolFromSmiles(smiles)
    new_smiles = general_ga_fixture._try_attach(mol)

    assert new_smiles is not None

    new_molecule = Chem.MolFromSmiles(new_smiles)
    assert is_linkers_ok(new_molecule, general_ga_fixture)


@pytest.mark.parametrize(
    "smiles_population",
    [
        [
            "CCC(C)(CC)C1C=CCC1C",
            "CC(C=C)=CCC=C(C#C)C#C",
            "CC1CC2(C)CCC(=C)CC2=C1",
            "CCC(C)C#CCCCC1CC1",
            "CC1CCC2CCCC(C2)CC1",
            "CC1CCC2C3CC(=C)C2C3C1",
            "CC=C1CC=C2CC=CCC12",
            "C=CC1(CC(=C)CC(=C)C1)C=C",
            "CCCC=CC(C)C(C)C#CC",
            "C=CC1CC#CC=CCCC=C1",
        ]
    ],
)
def test_initial_population(smiles_population, general_ga_fixture):
    molecules = general_ga_fixture.make_initial_population(
        smiles_population, randomize=False
    )
    for molecule in molecules:
        smiles = Chem.MolToSmiles(molecule)
        assert smiles is not None

        assert is_linkers_ok(molecule, general_ga_fixture)


@pytest.mark.parametrize(
    "smiles_population",
    [
        [
            "CCC(C)(CC)C1C=CCC1C",
            "CC(C=C)=CCC=C(C#C)C#C",
            "CC1CC2(C)CCC(=C)CC2=C1",
            "CCC(C)C#CCCCC1CC1",
            "CC1CCC2CCCC(C2)CC1",
            "CC1CCC2C3CC(=C)C2C3C1",
            "CC=C1CC=C2CC=CCC12",
            "C=CC1(CC(=C)CC(=C)C1)C=C",
            "CCCC=CC(C)C(C)C#CC",
            "C=CC1CC#CC=CCCC=C1",
            "CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1",
            "C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1",
            "Cc1ccc(S(=O)(=O)N2C(N)=C(C#N)C(c3ccc(Cl)cc3)C2C(=O)c2ccccc2)cc1",
            "CC(C#N)CNC(=O)c1cccc(Oc2cccc(C(F)(F)F)c2)c1",
        ]
    ],
)
def test_crossover(smiles_population, general_ga_fixture, general_crossover_fixture):
    molecules = general_ga_fixture.make_initial_population(
        smiles_population, randomize=False
    )
    correct_molecules = 0
    outer_tries, inner_tries = 6, 6
    total_tries = outer_tries * inner_tries
    for _ in range(outer_tries):
        mol1 = np.random.choice(molecules)
        mol2 = np.random.choice(molecules)

        for i in range(inner_tries):
            child = general_crossover_fixture.crossover(mol1, mol2)
            if child is not None:
                print(Chem.MolToSmiles(child))
                assert is_linkers_ok(child, general_ga_fixture)
                correct_molecules += 1
    assert correct_molecules > inner_tries
    assert False
