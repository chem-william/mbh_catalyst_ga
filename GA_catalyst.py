import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Union, Literal

import numpy as np
import numpy.typing as npt
# import submitit
from rdkit import Chem
from scipy.stats import rankdata
from tabulate import tabulate

from crossover import Crossover
import filters
import mutate as mu
# from catalyst import ts_scoring
from catalyst.utils import Individual
from sa import neutralize_molecules, sa_target_score_clipped

SLURM_SETUP = {
    "slurm_partition": "kemi1",
    "timeout_min": 30,
    "slurm_array_parallelism": 10,
}

class GA:
    def __init__(self,
        crossover: Crossover,
        population_size: int,
        seed_population: List[str],
        scoring_function,
        generations: int,
        mating_pool_size: int,
        mutation_rate: float,
        prune_population: bool,
        minimization: bool,
        selection_method: Union[Literal["rank"], Literal["roulette"]],
        selection_pressure: float,
        molecule_filters: List[Chem.Mol],
        path,
        random_seed: Optional[int] = None,
    ) -> None:
        # TODO: switch to use the new numpy rng api
        if random_seed:
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.crossover = crossover

        self.population_size = population_size
        self.seed_population = seed_population
        self.scoring_function = scoring_function
        self.generations = generations
        self.mating_pool_size = mating_pool_size
        self.mutation_rate = mutation_rate
        self.prune_population = prune_population
        self.minimization = minimization
        self.selection_method = selection_method
        self.selection_pressure = selection_pressure
        self.molecule_filters = molecule_filters
        self.path = path
    
    def slurm_scoring(self, sc_function, population, ids, cpus_per_task: int = 4, cleanup: bool = False):
        """Evaluates a scoring function for population on SLURM cluster

        Args:
            sc_function (function): Scoring function which takes molecules and id (int,int) as input
            population (List): List of rdkit Molecules
            ids (List of Tuples of Int): Index of each molecule (Generation, Individual)

        Returns:
            List: List of results from scoring function
        """
        executor = submitit.AutoExecutor(
            folder="scoring_tmp",
            slurm_max_num_timeout=0,
        )
        executor.update_parameters(
            name=f"sc_g{ids[0][0]}",
            cpus_per_task=cpus_per_task,
            slurm_mem_per_cpu="1GB",
            timeout_min=SLURM_SETUP["timeout_min"],
            slurm_partition=SLURM_SETUP["slurm_partition"],
            slurm_array_parallelism=SLURM_SETUP["slurm_array_parallelism"],
        )
        args = [cpus_per_task for p in population]
        jobs = executor.map_array(sc_function, population, ids, args)

        results = [
            catch(job.result, handle=lambda e: (np.nan, None)) for job in jobs
        ]  # catch submitit exceptions and return same output as scoring function (np.nan, None) for (energy, geometry)
        if cleanup:
            shutil.rmtree("scoring_tmp")
        return results
    
    def scoring(self, molecules: List[Chem.Mol], ids: List[int]) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        energies, structures = [], []
        for molecule, idx in zip(molecules, ids):
            tmp_result = self.scoring_function(molecule, idx)
            energies.append(tmp_result[0])
            structures.append(tmp_result[1])
        return np.array(energies), np.array(structures)

    def catch(func, *args, handle=lambda e: e, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            return handle(e)


    def make_initial_population(self, smiles: List[str], randomize: bool = True) -> List[Chem.Mol]:
        if randomize:
            sample = np.random.choice(smiles, self.population_size)
        else:
            sample = smiles[:self.population_size]
        
        population = []
        for smi in sample:
            mol_obj = Chem.MolFromSmiles(smi)
            if mol_obj is not None:
                population.append(mol_obj)
            else:
                print(f"Failed to add {smi} to initial population")

        return population

    def calculate_normalized_fitness(self, scores: npt.ArrayLike) -> npt.ArrayLike:
        normalized_fitness = scores / np.sum(scores)
        return normalized_fitness

    def calculate_fitness(self, scores: npt.ArrayLike) -> npt.ArrayLike:
        if self.minimization:
            scores *= -1

        if self.selection_method == "roulette":
            fitness = scores
        elif self.selection_method == "rank":
            scores[scores == np.nan] = -np.inf
            ranks = rankdata(scores, method="ordinal")
            n = len(ranks)
            if self.selection_pressure:
                fitness = np.array([
                    2 - self.selection_pressure + (2 * (self.selection_pressure - 1) * (rank - 1) / (n - 1))
                    for rank in ranks
                ])
            else:
                fitness = ranks / n
        else:
            raise ValueError(
                f"Only rank-based ('rank') or roulette ('roulette') selection are available, you chose {self.selection_method}."
            )

        return fitness


    def make_mating_pool(
        self, population: List[Individual], fitness: npt.ArrayLike
    ) -> List[Individual]:
        mating_pool = np.random.choice(
            population, p=fitness, size=self.mating_pool_size
        )
        return mating_pool


    def reproduce(
        self, mating_pool: List[Individual], generation: int
    ) -> List[Individual]:
        new_population = []
        counter = 0
        while len(new_population) < self.population_size:
            rand = np.random.random()
            if rand > self.mutation_rate:
                parent_A = np.random.choice(mating_pool)
                parent_B = np.random.choice(mating_pool)
                new_child = self.crossover.crossover(parent_A.rdkit_mol, parent_B.rdkit_mol)
                if new_child != None:
                    idx = (generation, counter)
                    counter += 1
                    new_child = Individual(rdkit_mol=new_child, idx=idx)
                    new_population.append(new_child)
            else:
                parent = random.choice(mating_pool)
                mutated_child = mu.mutate(parent.rdkit_mol, self.crossover, 1.0)
                if mutated_child != None:
                    idx = (generation, counter)
                    counter += 1
                    mutated_child = Individual(
                        rdkit_mol=mutated_child,
                        idx=idx,
                    )
                    new_population.append(mutated_child)
        return new_population


    def sanitize(self, population: List[Individual], prune_population: bool) -> List[Individual]:
        if prune_population:
            sanitized_population = []
            for ind in population:
                if ind.smiles not in [si.smiles for si in sanitized_population]:
                    sanitized_population.append(ind)
        else:
            sanitized_population = population

        sanitized_population.sort(
            key=lambda x: float("inf") if np.isnan(x.score) else x.score
        )  # np.nan is highest value, works for minimization of score

        new_population = sanitized_population[:self.population_size]
        return new_population  # selects individuals with lowest values


    def reweigh_scores_by_sa(
        self, population: List[Chem.Mol], scores: List[float]
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Reweighs scores with synthetic accessibility score

        population: list of RDKit molecules to be re-weighted
        scores: list of docking scores
        return: list of re-weighted docking scores
        """
        sa_scores = np.array([sa_target_score_clipped(p) for p in population])
        # return sa_scores, scores * sa_scores
        return sa_scores, scores
        # return sa_scores, [
        #     ns * sa for ns, sa in zip(scores, sa_scores)
        # ]  # rescale scores and force list type


    def print_results(self, population: List[Individual], fitness: npt.ArrayLike, generation: int):
        print(f"\nGeneration {generation+1}", flush=True)
        print(
            tabulate(
                [
                    [ind.idx, fit, ind.score, ind.energy, ind.sa_score, ind.smiles]
                    for ind, fit in zip(population, fitness)
                ],
                headers=[
                    "idx",
                    "normalized fitness",
                    "score",
                    "energy",
                    "sa score",
                    "smiles",
                ],
            ),
            flush=True,
        )

    def run(self):
        molecules = self.make_initial_population(self.seed_population, randomize=False)

        # write starting population
        generations_file = Path(self.path) / "generations.gen"
        with open(str(generations_file.resolve()), "w+") as fd:
            for m in molecules:
                # smi = Chem.MolToSmiles(m)
                fd.write(Chem.MolToSmiles(m))
            # f.writelines([Chem.MolToSmiles(m) + "\n" for m in molecules])

        ids = [(0, i) for i in range(len(molecules))]
        # results = self.slurm_scoring(self.scoring_function, molecules, ids)
        # energies = [res[0] for res in results]
        # geometries = [res[1] for res in results]
        # prescores = [energy - 100 for energy in energies]

        energies, geometries = self.scoring(molecules, ids)
        # prescores = energies - 100
        sa_scores, scores = self.reweigh_scores_by_sa(
            neutralize_molecules(molecules), energies
        )

        population = [
            Individual(
                idx=idx,
                rdkit_mol=mol,
                score=score,
                energy=energy,
                sa_score=sa_score,
                structure=structure,
            )
            for idx, mol, score, energy, sa_score, structure in zip(
                ids, molecules, scores, energies, sa_scores, geometries
            )
        ]
        population = self.sanitize(population, False)

        fitness = self.calculate_fitness(np.array([ind.score for ind in population]))
        fitness = self.calculate_normalized_fitness(fitness)

        self.print_results(population, fitness, -1)

        generations_list = []
        for generation in range(self.generations):
            print(f"Making mating pool..")
            mating_pool = self.make_mating_pool(population, fitness)

            print(f"Reproducing..")
            new_population = self.reproduce(
                mating_pool,
                generation + 1,
            )

            # new_resuls = slurm_scoring(
            #     scoring_function,
            #     [ind.rdkit_mol for ind in new_population],
            #     [ind.idx for ind in new_population],
            # )
            # new_energies = [res[0] for res in new_resuls]
            # new_geometries = [res[1] for res in new_resuls]
            print(f"Scoring..")
            new_energies, new_geometries = self.scoring(
                [ind.rdkit_mol for ind in new_population],
                [ind.idx for ind in new_population],
            )
            print(f"Reweighting..")
            new_sa_scores, new_scores = self.reweigh_scores_by_sa(
                neutralize_molecules([ind.rdkit_mol for ind in new_population]),
                new_energies,
            )

            for ind, score, energy, sa_score, structure in zip(
                new_population,
                new_scores,
                new_energies,
                new_sa_scores,
                new_geometries,
            ):
                ind.score = score
                ind.energy = energy
                ind.sa_score = sa_score
                ind.structure = structure

            print(f"Sanitizing..")
            population = self.sanitize(
                population + new_population, prune_population
            )

            print(f"Calculating fitness..")
            fitness = self.calculate_fitness(np.array([ind.score for ind in population]))
            fitness = self.calculate_normalized_fitness(fitness)

            generations_list.append([ind.idx for ind in population])
            self.print_results(population, fitness, generation)

        with open(str(generations_file.resolve()), "w+") as f:
            f.writelines(str(generations_list))

def test_scoring(molecule: Chem.Mol, idx: int) -> Tuple[int, int]:
    return 1/float(molecule.GetNumHeavyAtoms()), idx

if __name__ == "__main__":
    package_directory = Path(__file__).parent.resolve()

    population_size = 12
    # file_name = package_directory / "ZINC_amines.smi"
    scoring_function = test_scoring
    generations = 32
    mating_pool_size = population_size
    mutation_rate = 0.50
    seed_population = np.load("../../generate_molecules/gdb13/carbon_smiles.npz")["arr_0"]
    # seed_population = np.genfromtxt("../../generate_molecules/gdb13/7.smi", dtype="str")
    # seed_population = np.genfromtxt("./ZINC_amines.smi", dtype="str", autostrip=True)
    # with open("./ZINC_amines.smi") as fin:
    #     seed_population = [smiles for smiles in fin]

    print(seed_population)
    prune_population = True
    random_seed = 42
    minimization = True
    selection_method = "rank"
    selection_pressure = 1.5
    molecule_filters = filters.get_molecule_filters(
        ["MBH"], package_directory / "filters/alert_collection.csv"
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
    ga.run()
