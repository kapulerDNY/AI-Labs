import random
from deap import base, creator, tools, algorithms

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def eval_func(ind):
    x, y, z = ind
    result = 1 / (1 + (x - 2) ** 2 + (y + 1) ** 2 + (z - 1) ** 2)
    return result,

# Количество хромосом
num_bits = 3
toolbox = base.Toolbox()

toolbox.register("attr_float", random.uniform, -10, 10)#Float
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_bits)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxBlend, alpha=1)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=50)
algorithms.eaMuPlusLambda(population, toolbox, mu=20, lambda_=100, cxpb=0.7, mutpb=0.2, ngen=100)

best_ind = tools.selBest(population, 1)[0]
print("Лучший индивид: ", best_ind)
print("Максимум функции: ", best_ind.fitness.values[0])
