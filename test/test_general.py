from rdkit import Chem
import numpy as np
import pytest

import GA_catalyst
from ..GA_catalyst import GA

from crossover import Crossover
import filters


@pytest.fixture
def general_ga_fixture():
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
    molecule_filters = filters.get_molecule_filters(
        ["MBH"], "./filters/alert_collection.csv"
    )
    co = Crossover(average_size=8.0, size_stdev=4.0, molecule_filter=molecule_filters)

    ga = GA(
        crossover=co,
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
        molecule_filters=molecule_filters,
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
    new_smiles = general_ga_fixture._try_attach(mol, max_tries=5)

    assert new_smiles is not None

    new_molecule = Chem.MolFromSmiles(new_smiles)
    amount_tag = 0
    for atom in new_molecule.GetAtoms():
        if atom.GetSymbol() == general_ga_fixture.tagger_mol:
            amount_tag += 1
    assert amount_tag == 2
