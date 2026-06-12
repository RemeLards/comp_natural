from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sko.GA import RCGA 
import numpy as np
from sko.operators import selection
import matplotlib.pyplot as plt


class CustomRCGA(RCGA):

    def chrom2x(self, Chrom): # Sobrescrever a representação padrão
        return Chrom

iris = load_iris()
x_iris = iris.data       # características
y_iris = iris.target     # rótulos (apenas para avaliação)
GA_population_size = 6
max_it = 20


def fitness(chromosome):
    centroids = chromosome.reshape(3, 4)

    sse = 0.0

    for x in x_iris:
        dist_cen = np.sum((centroids - x) ** 2, axis=1)
        sse += np.min(dist_cen) # Pega a menor distância dos centros

    return sse


def crossover_arithmetic_GA(self):

    for i in range(0, self.size_pop, 2):

        alpha = np.random.rand()

        p1 = self.Chrom[i].copy()
        p2 = self.Chrom[i+1].copy()

        self.Chrom[i]   = alpha*p1 + (1-alpha)*p2
        self.Chrom[i+1] = alpha*p2 + (1-alpha)*p1

    return self.Chrom


def mutation_gaussian_GA(self, sigma=0.05):

    for i in range(self.size_pop):

        for j in range(self.n_dim):

            if np.random.rand() < self.prob_mut:

                self.Chrom[i,j] += np.random.normal(0, sigma)

    return self.Chrom


def Kmeans_explotation(population,best_fit_history):

    bf = None
    for i, chromosome in enumerate(population):
        centroids = chromosome.reshape(3,4)

        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        km = KMeans(
            init=centroids,
            n_clusters=3,
            n_init=1,
            max_iter=1
        )

        km.fit(x_iris)
        population[i] = km.cluster_centers_.flatten()
        fit = fitness(population[i])

        if not bf:
            bf = fit
        elif fit < bf:
            bf = fit
    best_fit_history.append(bf)    


def GA_Kmeans_Clusterization():
    GA_population = []
    best_fit_history = []
    bf = None
    for i in range(GA_population_size):
        kmeans = KMeans(
            init='random', #'k-means++', # Começa com centroídes distrubuídos baseados na inércia dos dados
            n_init=1, # Inicia N vezes e acha o melhor, mas no nosso caso como estamos usando um híbrido, não importa
            max_iter=1,
            n_clusters=3,
            random_state=100+i,
        )
        kmeans.fit(x_iris)
        # labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        GA_population.append(centroids.flatten())
        fit = fitness(centroids.flatten())
        if not bf:
            bf = fit
        elif fit < bf:
            bf = fit

    best_fit_history.append(bf)

    # https://scikit-opt.github.io/scikit-opt/#/en/more_ga
    n_clusters=3
    lb = np.tile(x_iris.min(axis=0), n_clusters)
    ub = np.tile(x_iris.max(axis=0), n_clusters)
    ga = CustomRCGA(
        func=fitness,
        size_pop=GA_population_size,
        n_dim=12,
        max_iter=1,
        lb=lb,
        ub=ub,
    )
    # https://scikit-opt.github.io/scikit-opt/#/en/README?id=feature1-udf
    ga.register(
        operator_name="selection",
        operator=selection.selection_tournament,
        tourn_size=2
    )
    ga.register(
        operator_name="crossover",
        operator=crossover_arithmetic_GA,
    )
    ga.register(
        operator_name="mutation",
        operator=mutation_gaussian_GA,
    )
    ga.Chrom = np.array(GA_population)
    for _ in range(max_it):
        best_x, best_y = ga.run()
        GA_population = ga.X.copy()
        Kmeans_explotation(GA_population,best_fit_history)
        ga.Chrom = GA_population
    # print(GA_population)
    # print('best_x:', best_x, '\n', 'best_y:', best_y)
    return best_fit_history


def Kmeans_Clusterization():
    fit_history = []
    kmeans = KMeans(
        init='random', #'k-means++', # Começa com centroídes distrubuídos baseados na inércia dos dados
        n_init=1, # Inicia N vezes e acha o melhor, mas no nosso caso como estamos usando um híbrido, não importa
        max_iter=1,
        n_clusters=3,
        random_state=100,
    )
    kmeans.fit(x_iris)
    centroids = kmeans.cluster_centers_
    fit_history.append(fitness(centroids.flatten()))
    for _ in range(max_it):
        kmeans = KMeans(
            init=centroids,
            max_iter=1,
            n_clusters=3,
            # random_state=42,
        )
        kmeans.fit(x_iris)
        centroids = kmeans.cluster_centers_
        fit_history.append(fitness(centroids.flatten()))
    
    return fit_history


def main():

    fit_history_KC = Kmeans_Clusterization()
    fit_history_GAKC = GA_Kmeans_Clusterization()

    plt.figure(figsize=(12, 4))

    plt.subplot(2, 1, 1)
    plt.plot(fit_history_KC)
    plt.title("K-Means")
    plt.xlabel("Iteração")
    plt.ylabel("SSE")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(fit_history_GAKC)
    plt.title("GA + K-Means")
    plt.xlabel("Geração")
    plt.ylabel("SSE")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
