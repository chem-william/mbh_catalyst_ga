from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pytest

import GA_catalyst
from ..GA_catalyst import GA

from crossover import Crossover
import filters


TAGGER_ATOM = "Se"
DUMMY_ATOM = "C"


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
        tagger_atom=TAGGER_ATOM,
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


def mutate(mol: Chem.Mol, rxn_smarts: str, co: Crossover) -> list[Chem.Mol]:
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)

    new_mol_trial = rxn.RunReactants((mol,))

    new_mols = []
    for m in new_mol_trial:
        m = m[0]
        new_mols.append(m)

    return new_mols


def compare_smiles(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # Generate the canonical SMILES strings for both molecules
    canonical_smiles1 = Chem.MolToSmiles(mol1)
    canonical_smiles2 = Chem.MolToSmiles(mol2)

    return canonical_smiles1 == canonical_smiles2


@pytest.mark.parametrize(
    "smiles, expected_result",
    [
        ("CCC", "CC"),
        (f"[{TAGGER_ATOM}]CCC", f"[{TAGGER_ATOM}]CC"),
    ],
)
def test_delete_atom_a(general_crossover_fixture, smiles, expected_result):
    from mutate import DeleteAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    reaction = mutate(mol, DeleteAtomChoices.a.value.replace("TAG", co.tagger_atom), co)
    assert compare_smiles(Chem.MolToSmiles(reaction[0]), expected_result)


@pytest.mark.parametrize(
    "smiles, expected_result",
    [
        ("CCCC", "CCC"),
        ("CC1([SeH])NC(C)([SeH])C1(C)", "CC1C(C)([SeH])C1(C)[SeH]"),
    ],
)
def test_delete_atom_b(general_crossover_fixture, smiles, expected_result):
    from mutate import DeleteAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    reaction = mutate(mol, DeleteAtomChoices.b.value.replace("TAG", co.tagger_atom), co)
    assert compare_smiles(Chem.MolToSmiles(reaction[0]), expected_result)


@pytest.mark.parametrize(
    "smiles, expected_result",
    [
        ("CC1([SeH])NC(C)([SeH])C1(N)", "CC1([SeH])NC(C)([SeH])N1"),
    ],
)
def test_delete_atom_c(general_crossover_fixture, smiles, expected_result):
    from mutate import DeleteAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    reaction = mutate(mol, DeleteAtomChoices.c.value.replace("TAG", co.tagger_atom), co)
    assert compare_smiles(Chem.MolToSmiles(reaction[0]), expected_result)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "CC1([SeH])NC(C)([SeH])C1(N)",
            [
                "CN1C(C)([SeH])C1(N)[SeH]",
                "CC1(N)N([SeH])C1(C)[SeH]",
                "CC1([SeH])C(N)N1C[SeH]",
                "CC1([SeH])NC1(N)C[SeH]",
                "CC1([SeH])C(N)CN1[SeH]",
                "CC1(N)N([SeH])C1(C)[SeH]",
                "CC1([SeH])NCC1(N)[SeH]",
                "CN1C(C)([SeH])C1(N)[SeH]",
                "CC1([SeH])NCC1(N)[SeH]",
                "CC1([SeH])NC1(N)C[SeH]",
                "CC1([SeH])NCC1(N)[SeH]",
                "CC1([SeH])NC1(N)C[SeH]",
                "CN1C(C)([SeH])C1(N)[SeH]",
                "CC1(N)N([SeH])C1(C)[SeH]",
                "CC1([SeH])C(N)CN1[SeH]",
                "CC1(N)N([SeH])C1(C)[SeH]",
                "CC1([SeH])C(N)N1C[SeH]",
                "CC1([SeH])NC1(N)C[SeH]",
                "CN1C(C)([SeH])C1(N)[SeH]",
                "CC1([SeH])NCC1(N)[SeH]",
                "CC1([SeH])C(N)N1C[SeH]",
                "CC1([SeH])C(N)CN1[SeH]",
                "CC1([SeH])C(N)CN1[SeH]",
                "CC1([SeH])C(N)N1C[SeH]",
            ],
        ),
    ],
)
def test_delete_atom_d(general_crossover_fixture, smiles, expected_results):
    from mutate import DeleteAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    reaction = mutate(mol, DeleteAtomChoices.d.value.replace("TAG", co.tagger_atom), co)
    for reac, expected in zip(reaction, expected_results):
        assert compare_smiles(Chem.MolToSmiles(reac), expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "CC1([SeH])N=CCC1(C)CC1CCC1",
            [
                "CC1(CC2CCC2)C=NC1(C)[SeH]",
                "CC1(CC2CCC2)C=NC1(C)[SeH]",
                "CC1([SeH])N=CCC1CC1CCC1",
                "CC1([SeH])N=CCC1CC1CCC1",
                "CC1([SeH])N=CCC1(C)C1CCC1",
                "CC1([SeH])N=CCC1(C)C1CCC1",
                "CC1(CC2CCC2)CC=NC1[SeH]",
                "CC1(CC2CCC2)CC=NC1[SeH]",
                "CC1(CC2CCC2)CC=NC1[SeH]",
                "CC1(CC2CCC2)CC=NC1[SeH]",
                "CC1([SeH])N=CCC1CC1CCC1",
                "CC1([SeH])N=CCC1CC1CCC1",
                "CC1([SeH])N=CCC1(C)C1CCC1",
                "CC1([SeH])N=CCC1(C)C1CCC1",
                "CC1(CC2CCC2)CC=NC1[SeH]",
                "CC1(CC2CCC2)CC=NC1[SeH]",
                "CC1(CC2CCC2)C=NC1(C)[SeH]",
                "CC1(CC2CCC2)C=NC1(C)[SeH]",
                "CC1([SeH])N=CCC1(C)C1CCC1",
                "CC1([SeH])N=CCC1(C)C1CCC1",
                "CC1(CC2CCC2)C=NC1(C)[SeH]",
                "CC1(CC2CCC2)C=NC1(C)[SeH]",
                "CC1([SeH])N=CCC1CC1CCC1",
                "CC1([SeH])N=CCC1CC1CCC1",
            ],
        ),
    ],
)
def test_delete_atom_e(general_crossover_fixture, smiles, expected_results):
    from mutate import DeleteAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    reaction = mutate(mol, DeleteAtomChoices.e.value.replace("TAG", co.tagger_atom), co)
    for reac, expected in zip(reaction, expected_results):
        assert compare_smiles(Chem.MolToSmiles(reac), expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "CC1([SeH])NC(C)([SeH])C1(N)",
            [
                "CC1([SeH])NC([SeH])(CC)C1N",
                "CC1([SeH])C(N)C(C)([SeH])N1C",
                "CC1([SeH])NC([SeH])(CC)C1N",
                "CC1([SeH])NC(C)([SeH])C1(N)C",
                "CC1([SeH])NC(C)([SeH])C1NC",
            ],
        ),
    ],
)
def test_append_atom_single(general_crossover_fixture, smiles, expected_results):
    from mutate import AppendAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = AppendAtomChoices.SINGLE.value.replace("X", "-" + DUMMY_ATOM)
    rxn_smarts = rxn_smarts.replace("TAG", TAGGER_ATOM)
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        assert compare_smiles(Chem.MolToSmiles(reac), expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "CC1([SeH])NC([SeH])C1N",
            [
                "C=CC1([SeH])NC([SeH])C1N",
                "C=NC1C([SeH])NC1(C)[SeH]",
            ],
        ),
    ],
)
def test_append_atom_double(general_crossover_fixture, smiles, expected_results):
    from mutate import AppendAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = AppendAtomChoices.DOUBLE.value.replace("X", "=" + DUMMY_ATOM)
    rxn_smarts = rxn_smarts.replace("TAG", TAGGER_ATOM)
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        s = Chem.MolToSmiles(reac)
        assert compare_smiles(s, expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "[SeH]CC([SeH])CCNCCCCC(Br)C",
            [
                "C#CC(Br)CCCCNCCC([SeH])C[SeH]",
            ],
        ),
    ],
)
def test_append_atom_triple(general_crossover_fixture, smiles, expected_results):
    from mutate import AppendAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = AppendAtomChoices.TRIPLE.value.replace("X", "#" + DUMMY_ATOM)
    rxn_smarts = rxn_smarts.replace("TAG", TAGGER_ATOM)
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        s = Chem.MolToSmiles(reac)
        assert compare_smiles(s, expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "CC1([SeH])NC(C)([SeH])C1(N)",
            [
                "CCC1([SeH])NC(C)([SeH])C1N",
                "CCC1([SeH])NC(C)([SeH])C1N",
                "CC1([SeH])NC(C)(C[SeH])C1N",
                "CC1([SeH])CNC(C)([SeH])C1N",
                "CC1([SeH])CC(N)C(C)([SeH])N1",
                "CC1([SeH])NC(C)(C[SeH])C1N",
                "CC1([SeH])CNC(C)([SeH])C1N",
                "CC1([SeH])CNC(C)([SeH])C1N",
                "CC1([SeH])CNC(C)([SeH])C1N",
                "CCC1([SeH])NC(C)([SeH])C1N",
                "CC1([SeH])NC(C)(C[SeH])C1N",
                "CC1([SeH])CC(N)C(C)([SeH])N1",
                "CCC1([SeH])NC(C)([SeH])C1N",
                "CC1([SeH])NC(C)(C[SeH])C1N",
                "CC1([SeH])CC(N)C(C)([SeH])N1",
                "CC1([SeH])NC(C)([SeH])C1CN",
                "CC1([SeH])CC(N)C(C)([SeH])N1",
                "CC1([SeH])NC(C)([SeH])C1CN",
            ],
        ),
    ],
)
def test_insert_atom_single(general_crossover_fixture, smiles, expected_results):
    from mutate import InsertAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = InsertAtomChoices.SINGLE.value.replace("X", DUMMY_ATOM)
    rxn_smarts = rxn_smarts.replace("TAG", TAGGER_ATOM)
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        s = Chem.MolToSmiles(reac)
        assert compare_smiles(s, expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "CC1([SeH])N=CCC1C1CCC1",
            [
                "C=CC1([SeH])N=CCC1C1CCC1",
                "CC1([SeH])NC=CCC1C1CCC1",
                "CC1([SeH])N=C=CCC1C1CCC1",
                "CC1([SeH])N=CC=CC1C1CCC1",
                "CC1([SeH])N=CC=CC1C1CCC1",
                "CC1([SeH])N=CCC=C1C1CCC1",
                "CC1([SeH])N=CCC1=CC1CCC1",
                "CC1([SeH])C=C(C2CCC2)CC=N1",
                "CC1([SeH])N=CCC1C=C1CCC1",
                "CC1([SeH])N=CCC1C1=CCCC1",
                "CC1([SeH])N=CCC1C1=CCCC1",
                "CC1([SeH])N=CCC1C1C=CCC1",
                "CC1([SeH])N=CCC1C1C=CCC1",
                "CC1([SeH])N=CCC1C1CC=CC1",
                "CC1([SeH])N=CCC1C1CC=CC1",
                "CC1([SeH])N=CCC1C1C=CCC1",
                "CC1([SeH])N=CCC1C1C=CCC1",
            ],
        ),
    ],
)
def test_insert_atom_double(general_crossover_fixture, smiles, expected_results):
    from mutate import InsertAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = InsertAtomChoices.DOUBLE.value.replace("X", DUMMY_ATOM)
    rxn_smarts = rxn_smarts.replace("TAG", TAGGER_ATOM)
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        s = Chem.MolToSmiles(reac)
        assert compare_smiles(s, expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "CC1([SeH])N=CCC1C1CCC1",
            [
                "C#CC1([SeH])N=CCC1C1CCC1",
            ],
        ),
    ],
)
def test_insert_atom_triple(general_crossover_fixture, smiles, expected_results):
    from mutate import InsertAtomChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = InsertAtomChoices.TRIPLE.value.replace("X", DUMMY_ATOM)
    rxn_smarts = rxn_smarts.replace("TAG", TAGGER_ATOM)
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        s = Chem.MolToSmiles(reac)
        assert compare_smiles(s, expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "C(#C)C1([SeH])N=CCC1C1CC=C1",
            [
                "CCC1([SeH])N=CCC1C1C=CC1",
                "CCC1([SeH])N=CCC1C1C=CC1",
                "C#CC1([SeH])NCCC1C1C=CC1",
                "C#CC1([SeH])NCCC1C1C=CC1",
                "C#CC1([SeH])N=CCC1C1CCC1",
                "C#CC1([SeH])N=CCC1C1CCC1",
            ],
        ),
    ],
)
def test_change_bond_order_tosingle(
    general_crossover_fixture, smiles, expected_results
):
    from mutate import ChangeBondOrderChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = ChangeBondOrderChoices.ToSingle.value.replace("TAG", TAGGER_ATOM)
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        s = Chem.MolToSmiles(reac)
        assert compare_smiles(s, expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "C(#C)C1([SeH])N=CCC1C1CC=C1",
            [
                "C#CC1([SeH])N=C=CC1C1C=CC1",
                "C#CC1([SeH])N=C=CC1C1C=CC1",
                "C#CC1([SeH])N=CC=C1C1C=CC1",
                "C#CC1([SeH])N=CC=C1C1C=CC1",
                "C#CC1([SeH])N=CCC1=C1C=CC1",
                "C#CC1([SeH])N=CCC1=C1C=CC1",
                "C#CC1([SeH])N=CCC1C1=CC=C1",
                "C#CC1([SeH])N=CCC1C1=C=CC1",
                "C#CC1([SeH])N=CCC1C1=CC=C1",
                "C#CC1([SeH])N=CCC1C1C=C=C1",
                "C#CC1([SeH])N=CCC1C1C=C=C1",
                "C#CC1([SeH])N=CCC1C1=C=CC1",
            ],
        ),
    ],
)
def test_change_bond_order_fromsingletodouble(
    general_crossover_fixture, smiles, expected_results
):
    from mutate import ChangeBondOrderChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = ChangeBondOrderChoices.FromSingleToDouble.value.replace(
        "TAG", TAGGER_ATOM
    )
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        s = Chem.MolToSmiles(reac)
        assert compare_smiles(s, expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "C(#C)C1([SeH])N=CCC1C1CC=C1",
            [
                "C=CC1([SeH])N=CCC1C1C=CC1",
                "C=CC1([SeH])N=CCC1C1C=CC1",
            ],
        ),
    ],
)
def test_change_bond_order_fromtripletodouble(
    general_crossover_fixture, smiles, expected_results
):
    from mutate import ChangeBondOrderChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = ChangeBondOrderChoices.FromTripleToDouble.value.replace(
        "TAG", TAGGER_ATOM
    )
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        s = Chem.MolToSmiles(reac)
        assert compare_smiles(s, expected)


@pytest.mark.parametrize(
    "smiles, expected_results",
    [
        (
            "C(=CCC)C1([SeH])N=CCC1C1CC=C1",
            [
                "C#CC=CC1([SeH])N=CCC1C1C=CC1",
                "C#CC=CC1([SeH])N=CCC1C1C=CC1",
            ],
        ),
    ],
)
def test_change_bond_order_totriple(
    general_crossover_fixture, smiles, expected_results
):
    from mutate import ChangeBondOrderChoices

    co = general_crossover_fixture

    mol = Chem.MolFromSmiles(smiles)
    rxn_smarts = ChangeBondOrderChoices.ToTriple.value.replace("TAG", TAGGER_ATOM)
    reaction = mutate(mol, rxn_smarts, co)
    for reac, expected in zip(reaction, expected_results):
        s = Chem.MolToSmiles(reac)
        assert compare_smiles(s, expected)
