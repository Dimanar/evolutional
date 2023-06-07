import pprint

import numpy
from deap import algorithms, base, creator, tools
import random

random.seed(23)

pack = [
    (9, 150, 'Карта'),
    (13, 35, 'Компас'),
    (153, 200, 'Вода'),
    (50, 160, 'Сендвіч'),
    (15, 60, 'Глюкоза'),
    (68, 45, 'Кружка'),
    (27, 60, 'Банан'),
    (39, 40, 'Яблуко'),
    (23, 30, 'Сир'),
    (52, 10, 'Пиво'),
    (11, 70, 'Крем від засмаги'),
    (32, 30, 'Камера'),
    (24, 15, 'Футболка'),
    (48, 10, 'Брюки'),
    (73, 40, 'Зонтик'),
    (42, 70, 'Водонепроникні штани'),
    (43, 75, 'Водонепроникний плащ'),
    (22, 80, 'Гаманець'),
    (7, 20, 'Сонцезахисні окуляри'),
    (18, 12, 'Рушник'),
    (4, 50, 'Носки'),
    (30, 10, 'Книга'),
]
PACK_SIZE = len(pack)
MAX_WEIGHT = 400
CROSSOVER_PROBABILITY = 0.4
MUTATION_PROBABILITY = 0.2
MAX_GENERATIONS = 120
POPULATION_SIZE = 35


def evaluate(individual):
    total_value = 0
    total_weight = 0
    for i in range(len(individual)):
        if individual[i] == 1:
            total_weight += pack[i][0]
            total_value += pack[i][1]

    if total_weight > MAX_WEIGHT:
        return 0,
    else:
        return total_value,


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / PACK_SIZE)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("zero_or_one", random.randint, 0, 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zero_or_one, n=PACK_SIZE)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)
stats.register("argmax", numpy.argmax)

toolbox.register("maximize", algorithms.eaSimple, toolbox=toolbox,
                 cxpb=CROSSOVER_PROBABILITY, mutpb=MUTATION_PROBABILITY, ngen=MAX_GENERATIONS,
                 stats=stats, verbose=True)

best_individuals, logbook = toolbox.maximize(population)

the_best_individual = logbook.select('argmax')[-1]
print('Список предметів, що треба взяти:')
pprint.pprint([name for using, (_, _, name) in zip(best_individuals[the_best_individual], pack) if using])
print('Загальна цінність взятих предметів =',
      sum([value for using, (_, value, _) in zip(best_individuals[the_best_individual], pack) if using]))
print('Загальна вага взятих предметів =',
      sum([weight for using, (weight, _, _) in zip(best_individuals[the_best_individual], pack) if using]))
