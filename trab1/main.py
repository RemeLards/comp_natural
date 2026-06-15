import copy

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.cluster import KMeans

from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.algorithm import Algorithm

from dataclasses import dataclass

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.replacement import ImprovementReplacement

from time import time


iris = load_iris()
x_iris = iris.data       # características
y_iris = iris.target     # rótulos (apenas para avaliação)
# iris_min_3_cluster = 78.8514

breast = load_breast_cancer()
x_breast = breast.data
y_breast = breast.target

wine = load_wine()
x_wine = wine.data
y_wine = wine.target

@dataclass
class FitnessHistory:
    iris : list = None
    breast: list = None
    wine: list = None


class CustomES_Elitism(ES): # -> Convergiu prematuramente
    # https://pymoo.org/algorithms/soo/es.html
    def _advance(self, infills=None, **kwargs):

        # Pais + filhos -> Lib seleciona somente os filhos (sem elitismo)
        infills = Population.merge(self.pop, infills)

        # Seleciona os μ melhores
        self.pop = self.survival.do(
            self.problem,
            infills,
            n_survive=self.pop_size
        )


class CustomES_Elitism_Kmeans(ES):
    # https://pymoo.org/algorithms/soo/es.html
    def _advance(self, infills=None, **kwargs):
        infills = Population.merge(self.pop, infills)
        pop = copy.deepcopy(infills.get("X"))
        kmeans_explotation(pop,self.problem.dataset_data,self.problem.n_clusters)
        infills.set("X", pop)

        self.evaluator.eval( # Utilizado para salvar histórico da fitness nova
            self.problem,
            infills
        )

        # Seleciona os μ melhores
        self.pop = self.survival.do(
            self.problem,
            infills,
            n_survive=self.pop_size
        )


class CustomES_Kmeans(ES):

    def _advance(self, infills=None, **kwargs):
        pop = copy.deepcopy(infills.get("X"))
        kmeans_explotation(pop,self.problem.dataset_data,self.problem.n_clusters)
        infills.set("X", pop)

        self.evaluator.eval(
            self.problem,
            infills
        )

        if len(infills) < self.pop_size:
            infills = Population.merge(infills, self.pop)

        self.pop = self.survival.do(self.problem, infills, n_survive=self.pop_size)


class CustomDE_Kmeans(DE):
    def _advance(self, infills=None, **kwargs):
        assert infills is not None, "This algorithms uses the AskAndTell interface thus infills must to be provided."

        pop = copy.deepcopy(infills.get("X"))
        kmeans_explotation(pop,self.problem.dataset_data,self.problem.n_clusters)
        infills.set("X", pop)

        self.evaluator.eval(
            self.problem,
            infills
        )

        # get the indices where each offspring is originating from
        pop_i = infills.get("index")
        # replace the individuals with the corresponding parents from the mating
        self.pop[pop_i] = ImprovementReplacement().do(self.problem, self.pop[pop_i], infills)

        # update the information regarding the current population
        FitnessSurvival().do(self.problem, self.pop, return_indices=True)


class DEPaper(Algorithm):

    def __init__(self,
                 CR,
                 pop_size=30,
                 tau=0.1,
                 L=0.05,
                 U=0.35,
                 Fmin=0.0,
                 Fmax=1.0,
                 **kwargs):

        super().__init__(**kwargs)

        self.pop_size = pop_size
        self.tau = tau
        self.L = L
        self.U = U
        self.Fmin = Fmin
        self.Fmax = Fmax
        if CR is None:
            raise ValueError("DEPaper needs a CR Value")
        self.CR = CR
        self.D = int(2/self.CR)


    def _initialize_infill(self):
        X = self._initialize_population()

        self.pop = Population.new(X=X)

        Fvec = np.random.uniform(
            self.Fmin,
            self.Fmax,
            self.pop_size
        )

        self.pop.set("Fvec", Fvec) # Vetor de F auto-adaptativos

        self.evaluator.eval(self.problem, self.pop)

        return self.pop
    

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = infills
        self.evaluator.eval(self.problem, self.pop)


    def _initialize_population(self):

        pop = np.zeros((self.pop_size, self.D))

        for i in range(self.pop_size):

            pop[i] = self._initialize_individual()

        return pop


    def _initialize_individual(self, t_kmeans=20):

        rng = np.random.default_rng(self.random_state)
        n_samples = self.problem.dataset_data.shape[0]

        # primeiro centróide aleatório escolhido dos dados
        idx = rng.integers(0, n_samples)
        centroids = [self.problem.dataset_data[idx]]

        # escolher os centroides demais
        for _ in range(1, self.problem.n_clusters):
            # amostra t pontos aleatórios (sem replacement)
            candidates_idx = rng.choice(n_samples, size=min(t_kmeans, n_samples), replace=False)
            candidates = self.problem.dataset_data[candidates_idx]

            # calcula distância mínima de cada candidato até centróides existentes
            dists = []
            for x in candidates:
                min_dist = np.min([np.linalg.norm(x - c) for c in centroids])
                dists.append(min_dist)

            # pega o mais distante
            best = candidates[np.argmax(dists)]
            centroids.append(best)

        return np.array(centroids).flatten()


    def _adapt_F(self):

        Fvec = self.pop.get("Fvec").copy()

        mask = np.random.rand(len(Fvec)) < self.tau

        Fvec[mask] = self.L + np.random.rand(np.sum(mask)) * (self.U - self.L)

        self.pop.set("Fvec", Fvec)


    def _mutation(self):

        X = self.pop.get("X")
        Fvec = self.pop.get("Fvec")

        mutant = np.empty_like(X)

        for i in range(self.pop_size):

            idx = np.arange(self.pop_size)

            idx = idx[idx != i]

            r1, r2, r3 = np.random.choice(
                idx,
                3,
                replace=False
            )

            mutant[i] = (
                X[r1]
                + Fvec[i] * (X[r2] - X[r3])
            )

        return mutant


    def _crossover(self, mutant):

        offspring = self.pop.get("X").copy()

        for i in range(self.pop_size):

            drand = np.random.randint(self.D)

            for j in range(self.D):

                if np.random.rand() <= self.CR or j == drand:

                    offspring[i, j] = mutant[i, j]

        return offspring
    

    def _kmeans_exploitation(self,population):
        for i, chromosome in enumerate(population):
            centroids = chromosome.reshape(self.problem.n_clusters,self.problem.dataset_data.shape[1])
            # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
            km = KMeans(
                init=centroids,
                n_clusters=self.problem.n_clusters,
                n_init=1,
                max_iter=1,
                tol=1e-4
            )

            km.fit(self.problem.dataset_data)
            population[i] = km.cluster_centers_.flatten()
        return np.array(population) 

    
    def _selection(self, temp_offspring):

        off = Population.new(X=temp_offspring)
        self.evaluator.eval(self.problem, off)

        off_X = off.get("X").copy()
        off_X = self._kmeans_exploitation(off_X)

        off.set("X", off_X)
        self.evaluator.eval(self.problem, off)

        # Atualizo se o indivíduo da população temporária for melhor que a versão não mutada+crossover
        X = self.pop.get("X")
        F = self.pop.get("F")
        X_off = off.get("X")
        F_off = off.get("F")

        F = np.array(F)
        F_off = np.array(F_off)

        replace = F_off[:, 0] < F[:, 0]
        
        X[replace] = X_off[replace]
        F[replace] = F_off[replace]


        # atualiza indivíduos e fitness
        self.pop.set("X", X)
        self.pop.set("F", F)


    def _advance(self, infills=None, **kwargs):

        self._adapt_F()

        mutant = self._mutation()
        temp_offspring = self._crossover(mutant)
        self._selection(temp_offspring)

        # reavaliar população atual
        self.evaluator.eval(self.problem, self.pop)


class ClusterProblem(Problem):
    def __init__(self,x,y):
        self.n_clusters = np.unique(y).size
        self.dataset_data = x
        super().__init__(
            n_var=self.n_clusters * self.dataset_data.shape[1],
            n_obj=1,
            xl=np.tile(x.min(axis=0), self.n_clusters),
            xu=np.tile(x.max(axis=0), self.n_clusters),
        )

    def _evaluate(self, x, out, *args, **kwargs):
        fitness_vec = np.zeros(len(x))
        for i, chromosome in enumerate(x):
            fitness_vec[i] = fitness(chromosome,self.dataset_data,self.n_clusters)

        out["F"] = fitness_vec



def kmeans_explotation(population,data, cluster_n):
    for i, chromosome in enumerate(population):
        centroids = chromosome.reshape(cluster_n,data.shape[1])
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        km = KMeans(
            init=centroids,
            n_clusters=cluster_n,
            n_init=1,
            max_iter=1
        )

        km.fit(data)
        population[i] = km.cluster_centers_.flatten()


def fitness(chromosome, data, cluster_n):
    centroids = chromosome.reshape(cluster_n, data.shape[1])

    sse = 0.0

    for x in data:
        dist_cen = np.sum((centroids - x) ** 2, axis=1)
        sse += np.min(dist_cen) # Pega a menor distância dos centros

    return round(sse, 4) # 4 Casas decimais


def RunPymooProblem(
    algorithm,
    problem : ClusterProblem,
    termination : tuple,
    seed: int | None,
    save_history : bool = True,
):
    if not seed:
        seed = 100
    res = minimize(
        problem=problem,
        algorithm=algorithm,
        termination=termination,
        seed=seed,
        verbose=True,
        save_history=save_history
    )
    
    print(res.X)   # melhor cromossomo (12 valores)
    print(res.F)   # melhor SSE
    # print("history:", res.history)
    # print("len(history):", len(res.history) if res.history is not None else None)
    # if res.history:
    #     print("history[0]:", res.history[0].opt.get("F")[0][0])
    return res


def ESClusterizationProblems(
    seed: int | None,
    es_class,
    max_it: int = 80,
):
    iris_problem = ClusterProblem(
        x=x_iris,
        y=y_iris,
    )
    breast_problem = ClusterProblem(
        x=x_breast,
        y=y_breast,
    )
    wine_problem = ClusterProblem(
        x=x_wine,
        y=y_wine,
    )
    iris = RunPymooProblem(
        algorithm=es_class(
            pop_size=30,
            rule=1/7, # pais/filhos = sigma/lambda = 1/7
            gamma=None,
        ),
        problem=iris_problem,
        termination=("n_iter", max_it),
        # termination=TerminationCollection(
        #     MaximumGenerationTermination(max_it),
        #     MinimumFunctionValueTermination(iris_min_3_cluster)
        # ),
        seed=seed,
    )
    breast = RunPymooProblem(
        algorithm=es_class(
            pop_size=30,
            rule=1/7, # pais/filhos = sigma/lambda = 1/7
            gamma=None,
        ),
        problem=breast_problem,
        termination=("n_iter", max_it),
        seed=seed,
    )
    wine = RunPymooProblem(
        algorithm=es_class(
            pop_size=30,
            rule=1/7, # pais/filhos = sigma/lambda = 1/7
            gamma=None,
        ),
        problem=wine_problem,
        termination=("n_iter", max_it),
        seed=seed,
    )
    

    return FitnessHistory(
        iris=iris.history,
        breast=breast.history,
        wine=wine.history,
    )


def DEClusterizationProblems(
    de_class,
    max_it: int = 80,
    seed: int | None = None,
):
    #https://pymoo.org/algorithms/soo/de.html
    iris_problem = ClusterProblem(
        x=x_iris,
        y=y_iris,
    )
    breast_problem = ClusterProblem(
        x=x_breast,
        y=y_breast,
    )
    wine_problem = ClusterProblem(
        x=x_wine,
        y=y_wine,
    )
    iris = RunPymooProblem(
        algorithm=de_class(
            variant="DE/best/1/bin",
            pop_size=200,
            CR=2.0/float((np.unique(y_iris).size * x_iris.shape[1])),
            F=0.3,
        ),
        problem=iris_problem,
        termination=("n_iter", max_it),
        seed=seed,
    )
    breast = RunPymooProblem(
        algorithm=de_class(
            variant="DE/best/1/bin",
            pop_size=200,
            CR=2.0/float((np.unique(y_breast).size * x_breast.shape[1])),
            F=0.3,
        ),
        problem=breast_problem,
        termination=("n_iter", max_it),
        seed=seed,
    )
    wine = RunPymooProblem(
        algorithm=de_class(
            variant="DE/best/1/bin",
            pop_size=200,
            CR=2.0/float((np.unique(y_wine).size * x_wine.shape[1])),
            F=0.3,
        ),
        problem=wine_problem,
        termination=("n_iter", max_it),
        seed=seed,
    )

    
    return FitnessHistory(
        iris=iris.history,
        breast=breast.history,
        wine=wine.history,
    )


def plot_results_table(fitness_dict):
    data = []
    for algorithm, histories in fitness_dict.items():

        if len(histories) == 0:
            continue

        # percorre iris, breast e wine separadamente
        for dataset_name in ["iris", "breast", "wine"]:

            best_runs = []
            iter_runs = []
            # print(histories)
            for fitness_history in histories:

                dataset = getattr(fitness_history, dataset_name)

                if dataset is None:
                    continue

                best = min(run.opt.get("F")[0][0] for run in dataset)
                best_runs.append(best)
                iter_runs.append(len(dataset))

            if len(best_runs) == 0:
                continue

            data.append([
                algorithm,
                dataset_name,
                # f"{np.mean(iter_runs):.1f}",
                f"{np.min(best_runs):.4f}",
                f"{np.mean(best_runs):.4f}",
            ])

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(data) + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=data,
        colLabels=[
            "Algoritmo",
            "Dataset",
            "Melhor resultado",
            "Resultado médio"
        ],
        loc="center",
        cellLoc="center"
    )

    best_per_dataset = {}

    for row in data:
        dataset = row[1]
        best = float(row[2])
        mean = float(row[3])

        if dataset not in best_per_dataset:
            best_per_dataset[dataset] = [best, mean]
        else:
            best_per_dataset[dataset][0] = min(best_per_dataset[dataset][0], best)
            best_per_dataset[dataset][1] = min(best_per_dataset[dataset][1], mean)

    for i, row in enumerate(data, start=1):

        dataset = row[1]
        best = float(row[2])
        mean = float(row[3])

        if best == best_per_dataset[dataset][0]:
            table[(i, 2)].get_text().set_fontweight("bold")

        if mean == best_per_dataset[dataset][1]:
            table[(i, 3)].get_text().set_fontweight("bold")

    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(1000) # Para facilmente ser reproduzível
    seeds = np.random.randint(0, 2**32 - 1, size=5)
    # print(seeds)
    # fitness_es_dict = {}
    # fitness_es_dict["es"] = []
    # fitness_es_dict["es-elitism"] = []
    # fitness_es_dict["es-kmeans"] = []
    # fitness_es_dict["es-elitism-kmeans"] = []
    # for seed in np.random.randint(0, 2**32 - 1, size=2):
    #     fitness_es = ESClusterizationProblems(seed=seed,es_class=ES)
    #     fitness_es_kmeans = ESClusterizationProblems(seed=seed,es_class=CustomES_Kmeans)
    #     fitness_es_elitism = ESClusterizationProblems(seed=seed,es_class=CustomES_Elitism)
    #     fitness_es_elistism_kmeans = ESClusterizationProblems(seed=seed,es_class=CustomES_Elitism_Kmeans)

    #     fitness_es_dict["es"].append(fitness_es)
    #     fitness_es_dict["es-kmeans"].append(fitness_es_kmeans)
    #     fitness_es_dict["es-elitism"].append(fitness_es_elitism)
    #     fitness_es_dict["es-elitism-kmeans"].append(fitness_es_elistism_kmeans)
    # plot_results_table(fitness_es_dict)

    fitness_de_dict = {}
    fitness_de_dict["de"] = []
    fitness_de_dict["de-kmeans"] = []
    fitness_de_dict["de-paper"] = []
    for i,seed in enumerate(seeds):
        it_start_time = time()
        print(f"Running Seed {i+1} : {seed}")
        # fitness_de = DEClusterizationProblems(seed=seed, de_class=DE)
        # fitness_de_kmeans = DEClusterizationProblems(seed=seed, de_class=CustomDE_Kmeans)
        fitness_de_paper = DEClusterizationProblems(seed=seed,de_class=DEPaper)

        # fitness_de_dict["de"].append(fitness_de)
        # fitness_de_dict["de-kmeans"].append(fitness_de_kmeans)
        fitness_de_dict["de-paper"].append(fitness_de_paper)
        print(f"Finished Seed {i+1}: {seed}, Runtime : {time()-it_start_time:.2f}s")
    plot_results_table(fitness_de_dict)
    


if __name__ == "__main__":
    main()
