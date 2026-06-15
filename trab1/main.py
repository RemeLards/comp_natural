import copy

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.cluster import KMeans

from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.fmin import MinimumFunctionValueTermination
from pymoo.termination.collection import TerminationCollection

from dataclasses import dataclass
from functools import partial

from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import FitnessSurvival
from pymoo.core.replacement import ImprovementReplacement
from pymoo.core.sampling import Sampling


iris = load_iris()
x_iris = iris.data       # características
y_iris = iris.target     # rótulos (apenas para avaliação)
iris_min_3_cluster = 78.8514

breast = load_breast_cancer()
x_breast = breast.data
y_breast = breast.target

wine = load_wine()
x_wine = wine.data
y_wine = wine.target

DE_population_size = 30

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
        I = infills.get("index")
        # replace the individuals with the corresponding parents from the mating
        self.pop[I] = ImprovementReplacement().do(self.problem, self.pop[I], infills)

        # update the information regarding the current population
        FitnessSurvival().do(self.problem, self.pop, return_indices=True)


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


class KMeansRandomSampling(Sampling):

    def __init__(self, data, cluster_n):
        super().__init__()
        self.data = data
        self.cluster_n = cluster_n

    def _do(self, problem, n_samples, **kwargs):

        x = []

        for _ in range(n_samples):

            km = KMeans(
                init='random',
                n_clusters=self.cluster_n,
                n_init=1,
                max_iter=1
            )

            km.fit(self.data)

            x.append(km.cluster_centers_.flatten())

        return np.asarray(x)


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
    
    # print(res.X)   # melhor cromossomo (12 valores)
    # print(res.F)   # melhor SSE
    # print("history:", res.history)
    # print("len(history):", len(res.history) if res.history is not None else None)
    # if res.history:
    #     print("history[0]:", res.history[0].opt.get("F")[0][0])
    return res


def ESClusterizationProblems(
    seed: int | None,
    es_class,
    max_it: int = 200,
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
    max_it: int = 200,
    seed: int | None = None,
    sampling: Sampling | None = None,
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
    if not sampling:
        iris = RunPymooProblem(
            algorithm=de_class(
                variant="DE/best/1/bin",
                pop_size=200,
                CR=0.2,
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
                CR=0.2,
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
                CR=0.2,
                F=0.3,
            ),
            problem=wine_problem,
            termination=("n_iter", max_it),
            seed=seed,
        )
    else:
        iris = RunPymooProblem(
            algorithm=de_class(
                variant="DE/best/1/bin",
                pop_size=200,
                CR=0.2,
                F=0.3,
                sampling=sampling,
            ),
            problem=iris_problem,
            termination=("n_iter", max_it),
            seed=seed,
        )
        breast = RunPymooProblem(
            algorithm=de_class(
                variant="DE/best/1/bin",
                pop_size=200,
                CR=0.2,
                F=0.3,
                sampling=sampling,
            ),
            problem=breast_problem,
            termination=("n_iter", max_it),
            seed=seed,
        )
        wine = RunPymooProblem(
            algorithm=de_class(
                variant="DE/best/1/bin",
                pop_size=200,
                CR=0.2,
                F=0.3,
                sampling=sampling,
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
            print(histories)
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

        if np.isclose(best, best_per_dataset[dataset][0]):
            table[(i, 2)].get_text().set_fontweight("bold")

        if np.isclose(mean, best_per_dataset[dataset][1]):
            table[(i, 3)].get_text().set_fontweight("bold")

    plt.tight_layout()
    plt.show()


def DE_Kmeans_Clusterization(data, cluster_n, max_it, seed):
    #https://scikit-opt.github.io/scikit-opt/#/en/README?id=_1-differential-evolution
    DE_population = []
    # best_fit_history = []

    fitness_func = partial( # Reaproveito a função feita para o Pymoo
        fitness,
        data=data,
        cluster_n=cluster_n
    )

    bf = None

    # população inicial via KMeans
    for i in range(DE_population_size):

        km = KMeans(
            init='random', #'k-means++', # Começa com centroídes distrubuídos baseados na inércia dos dados
            n_clusters=cluster_n, # Inicia N vezes e acha o melhor, mas no nosso caso como estamos usando um híbrido, não importa
            n_init=1,
            max_iter=1,
            random_state=seed+i
        )

        km.fit(data)

        chromosome = km.cluster_centers_.flatten()

        DE_population.append(chromosome)

        fit = fitness_func(chromosome)

        if bf is None or fit < bf:
            bf = fit

    # best_fit_history.append(bf)

    lb = np.tile(data.min(axis=0), cluster_n)
    ub = np.tile(data.max(axis=0), cluster_n)

    de = DE(
        func=fitness_func,
        n_dim=cluster_n * data.shape[1],
        size_pop=DE_population_size,
        max_iter=1,
        lb=lb,
        ub=ub,
        F=0.1,
        prob_mut=0.03
    )

    # substitui população inicial
    # https://scikit-opt.github.io/scikit-opt/#/en/more_ga
    de.X = np.array(DE_population)

    for _ in range(max_it):

        best_x, best_y = de.run()

        DE_population = de.X.copy()

        kmeans_explotation(
            DE_population,
            data,
            cluster_n
        )

        de.X = DE_population

        bf = min(fitness_func(ind) for ind in DE_population)
        # best_fit_history.append(bf)

    print(best_x)
    print(best_y)

    return best_x,best_y


def main():
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
    fitness_de_dict["de-kmeans-sampling"] = []
    for seed in np.random.randint(0, 2**32 - 1, size=1):
        fitness_de = DEClusterizationProblems(seed=seed, de_class=DE)
        fitness_de_kmeans = DEClusterizationProblems(seed=seed, de_class=CustomDE_Kmeans)

        fitness_de_dict["de"].append(fitness_de)
        fitness_de_dict["de-kmeans"].append(fitness_de_kmeans)
    plot_results_table(fitness_de_dict)
    

    # fit_history_KC = Kmeans_Clusterization()
    # fit_history_GAKC = GA_Kmeans_Clusterization()

    # plt.figure(figsize=(12, 4))

    # plt.subplot(2, 1, 1)
    # plt.plot(fit_history_KC)
    # plt.title("K-Means")
    # plt.xlabel("Iteração")
    # plt.ylabel("SSE")
    # plt.grid(True)

    # plt.subplot(2, 1, 2)
    # plt.plot(fit_history_GAKC)
    # plt.title("GA + K-Means")
    # plt.xlabel("Geração")
    # plt.ylabel("SSE")
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
