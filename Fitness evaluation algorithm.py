import numpy as np
import random
from prettytable import PrettyTable

table = PrettyTable()

# start with an easy problem maximising f(x) = x^2 for some x [0,31]
# clearly the answer is x = 31
# we encode data into five-string binary bbbbb for b in {0,1} therefore bounded by
# 0 and 2^4 + 2^3 + 2^2 + 2^1 + 2^0 = 31

# lets make a binary encoder and decoder first

def bin_decoder(b):
    x = []
    for i in range(len(b)):
        a = b[i]* (2 ** (4-i))
        x.append(a)
    return int(np.sum(x))


def bin_encoder(x):
    i = 4
    b = []
    while i >= 0:
        if 2**i <= x:
            b.append(1)
            x-=2**i
            i-=1
        
        else:
            b.append(0)
            i-=1
    return b

# now that we can work with decimal and binary number systems interchangably, we will make a fitness evaluation function
# f(x) = x^2
def fitness_eval_bin(b):
    f = bin_decoder(b) ** 2
    return f

def fitness_eval_dec(x):
    return x**2

# Next we need probability of each specimen to be selected for mating
# Lets set up our population first
N = 4
pop = random.sample(range(0,31),N)
pop.sort()

def population_fitnesses(population):
    fitnesses = []
    for p in population:
        a = fitness_eval_dec(p)
        fitnesses.append(a)

    total_fitness = sum(fitnesses)
    return fitnesses, total_fitness

a,b = population_fitnesses(pop)

print("Population: ",pop)
print("Population fitnesses: ", a)
print("Total fitness: ", b)

def selection(population):
    fitnesses, total_fitness = population_fitnesses(population)
    cumulative_fitnesses = [0]
    c = 0
    for f in fitnesses:
        a  = f / total_fitness
        c += a
        cumulative_fitnesses.append(c)
    print("Cumulative fitnesses: ", cumulative_fitnesses)
    selected = []
    for i in range(4):
        r = random.random()
        print('r_%d ' % i, r)
        for j in range(len(cumulative_fitnesses)):
            if cumulative_fitnesses[j] <= r and r < cumulative_fitnesses[j+1]:
                selected.append(population[j])
    return selected

selected = selection(pop)

print("Selected parents: ", selected)

selected_bin = [bin_encoder(s) for s in selected]
print("Selected parents binary:", selected_bin)

def crossover(population):
    population_bin = [bin_encoder(p) for p in population]
    offspring = []
    for i in range(int(len(population_bin)/2)):
        p1 = population_bin[2 * i]
        print("Parent 1: ", p1)
        p2 = population_bin[2 * i + 1]
        print("Parent 2: ", p2)
        crs_pnt = random.randint(0,3)
        print("Crossover point: ", crs_pnt)
        g11 = p1[0:crs_pnt+1]
        g21 = p2[crs_pnt + 1: 5]
        print("gene 1, child 1: ", g11,"gene 2, child 1: ",g21)
        c1 = g11 + g21
        offspring.append(c1)

        g12 = p1[crs_pnt+1:5]
        g22 = p2[0:crs_pnt+1]
        print("gene 1, child 2: ", g12, "gene 2, child 2: ", g22)
        c2 = g22 + g12
        offspring.append(c2)
    
    offspring_dec = [bin_decoder(o) for o in offspring]

    return offspring, offspring_dec

offspring, offspring_dec = crossover(selected)
print("Offspring: ", offspring)
print("Offspring dec: ", offspring_dec)

offspring_fitness = [fitness_eval_dec(a) for a in offspring_dec]
total_offspring_fitness = sum(offspring_fitness)

print("Initial population fitness: ", b)
print("Offsping fitness:", total_offspring_fitness)


table.add_column("Initial population", pop)
table.add_column("Initial fitness", a)
table.add_column("Offspring", offspring_dec)
table.add_column("Offspring fitness", offspring_fitness)
print(table)


# references
# https://engineering.purdue.edu/~sudhoff/ee630/Lecture02.pdf