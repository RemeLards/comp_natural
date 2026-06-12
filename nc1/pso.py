import numpy as np
from sko.PSO import PSO
import matplotlib.pyplot as plt

N_PARTICLES=20
MAX_ITERATIONS=70
N_RUNS=10
SEED=100 #importante para replicabilidade

#random seed
np.random.seed(SEED)

# carregar dados
x = np.loadtxt("x_data.txt")
y = np.loadtxt("y_data.txt")

# modelo que queremos ajustar
def model(x, a, b, c):
    return a * x**2 + b * x + c


# função objetivo (SSE)
def sse(params):
    a, b, c = params
    y_pred = model(x, a, b, c)
    return np.sum((y - y_pred) ** 2)


# limites dos parâmetros
lb = [-5, -5, -5]
ub = [5, 5, 5]

# armazenar resultados
all_params = []
all_sse = []

x_plot = np.linspace(min(x), max(x), 200)

fig, axes = plt.subplots(2, 5, figsize=(12, 12))
axes = axes.flatten()
plt.scatter(x, y, s=10, color='red', label='data')
colors = plt.cm.viridis(np.linspace(0, 1, N_RUNS))

for i in range(N_RUNS):

    pso = PSO(
        func=sse,
        n_dim=3,
        pop=N_PARTICLES,
        max_iter=MAX_ITERATIONS,
        lb=lb,
        ub=ub,
        w=0.8, # w < 1
        c1=1.5, # c1 E [1,3]
        c2=1.5, # c2 E [1,3]
    )

    pso.run()

    best_params = pso.gbest_x
    best_sse = pso.gbest_y.item()

    all_params.append(best_params)
    all_sse.append(best_sse)

    y_plot = model(x_plot, *best_params)

    ax = axes[i]

    ax.scatter(x, y, s=10, color='red')
    ax.plot(x_plot, y_plot, color=colors[i], linewidth=2)
    ax.set_title(f"PSO {i+1}")
    ax.text(
        0.05, 0.95,
        f"a={best_params[0]:.3f}\n"
        f"b={best_params[1]:.3f}\n"
        f"c={best_params[2]:.3f}\n"
        f"SSE={best_sse:.3f}",
        transform=ax.transAxes,
        va='top',
        fontsize=8,
        bbox=dict(facecolor='white', alpha=0.7)
    )

plt.tight_layout()


all_params = np.array(all_params)
all_sse = np.array(all_sse)
table_data = [
    ["Melhor SSE", round(all_sse.min(),4)],
    ["SSE Médio", round(np.mean(all_sse), 4)],
    ["Melhores parâmetros (a,b,c)", np.round(all_params[np.argmin(all_sse)], 4)],
]

fig, ax = plt.subplots(figsize=(6, 3))
table = ax.table(
    cellText=table_data,
    colLabels=["Métrica", "Valores"],
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

ax.axis('off')
plt.title("PSO Estatísticas")
plt.show()