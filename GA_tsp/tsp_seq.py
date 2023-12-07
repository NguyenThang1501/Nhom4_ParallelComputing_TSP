import math
import time

import matplotlib.pyplot as plt
import numpy as np


def calculate_distance(distance_matrix, order):
    # calculates the distance between the cities in the order given
    distance = 0
    for i in range(len(order) - 1):
        distance += distance_matrix[order[i]][order[i + 1]]
    distance += distance_matrix[order[len(order) - 1]][order[0]]
    return distance


def swap(arr, i, j):
    # swaps the cities at two locations in our tour
    arr[i], arr[j] = arr[j], arr[i]


def create_population(population, P_SIZE, order, fitness):
    # create a population of random orders
    for i in range(P_SIZE):
        new_order = order.copy()
        np.random.shuffle(new_order)
        population.append(new_order)
        fitness.append(1.0)


def normalize_fitness(fitness, P_SIZE):
    # normalize the fitness values
    sm = sum(fitness)
    for i in range(P_SIZE):
        fitness[i] = fitness[i] / sm


def calculate_fitness(distance_matrix, population, fitness):
    # function to calculate the fitness of each individual in the population
    for i in range(len(population)):
        order = population[i]
        d = calculate_distance(distance_matrix, order)
        fitness[i] = 1 / (d + 1)


def pick_one(population, prob):
    # picks one individual from the population based on the probability
    index = 0
    r = np.random.rand()
    np.random.shuffle(population)
    while r > 0:
        r -= prob[index]
        index += 1
    index -= 1
    return population[index].copy()


def mutate(distance_matrix, order, P_SIZE, iters, mutation_rate=1, is_best=False):
    # function to mutate the order of cities
    n = len(order)  # number of cities
    # based on the mutation rate, the order of the cities is changed by swapping the cities
    if iters % 7 == 0:  # If the current iteration is divisible by 7, the function randomly selects a city (`j`) and another city (`k`) at a distance of `m` cities from `j`.
        # The cities at positions `j` and `k` are then swapped using the `swap` function.
        for i in range(mutation_rate):
            j = np.random.randint(0, len(order))
            m = np.random.randint(1, int(n / 2))
            k = (j + m) % n
            swap(order, k, j)
    else:  # If the current iteration is not divisible by 7, the function randomly selects a city (`j`) and another city (`k`) next to `j`.
        j = np.random.randint(0, len(order))
        m = 1
        k = (j + m) % n
        swap(order, k, j)
    if iters % 12 == 0 and mutation_rate != 0:  # If the current iteration is divisible by 12 and the mutation rate is not zero, the function shuffles the entire order randomly.
        np.random.shuffle(order)
        return order
    #  If the current iteration is divisible by 10 (and it is the best individual)
    #  or if the current iteration is divisible by 69 and a random number between 0 and 100 is divisible by 7,
    #  the function performs a partial reversal mutation.
    if (iters % 10 == 0 and is_best) or (iters % 69 == 0 and np.random.randint(0, 100) % 7 == 0):
        d = calculate_distance(distance_matrix, order)  # calculates the distance of the current order
        p = np.random.randint(0, n)  # selects a random city
        for i in range(p, p + math.ceil(n / 2)):
            i = i % n  # wraps around the order array
            if np.random.randint(0, 100) % 2 == 0:  # randomly selects whether to reverse the order or not
                for j in reversed(range(math.ceil(n / 2))):  # reverses the order of the cities
                    k = (i + j) % n
                    swap(order, k, j)
                    temp = calculate_distance(distance_matrix, order)  # calculates the distance of the new order
                    if temp < d:  # if the new order is better than the previous order, the new order is retained
                        d = temp  # updates the distance
                        if iters < 100 + int(1000 / P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return order
                    else:
                        swap(order, k, j)
            else:  # if the order is not reversed
                for j in range(math.ceil(n / 2)):
                    k = (i + j) % n
                    swap(order, k, j)
                    temp = calculate_distance(distance_matrix, order)
                    if temp < d:
                        d = temp
                        if iters < 100 + int(1000 / P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return order
                    else:
                        swap(order, k, j)

    if ((iters + 1) % 15 == 0 and is_best and iters % 5 != 0) or (
            iters < 100 and iters % 5 == 0):  # the function performs a partial reversal mutation.
        d = calculate_distance(distance_matrix, order)
        p = np.random.randint(0, n)
        for i in range(p, p + n):
            i = i % n
            if np.random.randint(0, 100) % 2 == 0:
                for j in reversed(range(int(n / 2))):
                    k = (i + j) % n
                    if k < i:
                        temp = i
                        i = k
                        k = temp
                    order[i:k + 1] = np.flip(order[i:k + 1])
                    temp = calculate_distance(distance_matrix, order)
                    if temp < d:
                        d = temp
                        if iters < 100 + int(1000 / P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return order
                    else:
                        order[i:k + 1] = np.flip(order[i:k + 1])
            else:
                for j in range(int(n / 2)):
                    k = (i + j) % n
                    if k < i:
                        temp = i
                        i = k
                        k = temp
                    order[i:k + 1] = np.flip(order[i:k + 1])
                    temp = calculate_distance(distance_matrix, order)
                    if temp < d:
                        d = temp
                        if iters < 100 + int(1000 / P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return order
                    else:
                        order[i:k + 1] = np.flip(order[i:k + 1])
    return order


def cross_over(order1, order2, iters):
    # function to perform cross over between two orders
    n = len(order1)  # number of cities
    i1 = np.random.randint(0, n)  # selects a random city
    i2 = np.random.randint(0, n)  # selects another random city

    order = np.zeros(n)
    if iters % 2 == 0 or iters % 3 == 0:  # if the current iteration is divisible by 2 or 3, the order of the cities is reversed
        order1 = np.flip(order1)

    order[i1:i2 + 1] = order1[i1:i2 + 1]  # copies the order of the cities from the first order
    order[:i1] = -1  # fills the rest of the order with -1
    order[i2 + 1:] = -1  # fills the rest of the order with -1
    # fills the rest of the order with the cities from the second order
    for i in range(n):
        if order[i] == -1:
            for j in range(n):
                if order2[j] not in order:
                    order[i] = order2[j]
                    break
    return np.array(order, dtype=np.int32).tolist()


def next_generation(distance_matrix, population, P_SIZE, fitness, iters, best=None):
    # creates the next generation of the population
    next_gen = []  # stores the next generation
    n = P_SIZE
    if str(type(best)) != str(type(None)):  # if the best individual is not None, it is added to the next generation
        next_gen.append(best)
        n -= 1

    for i in range(n):
        m_rate = np.random.randint(1, 3)
        order3 = pick_one(population, fitness)  # selects an individual from the population
        order3 = mutate(distance_matrix, order3, P_SIZE, iters, m_rate)  # mutates the order of the cities

        order1 = pick_one(population, fitness)
        order2 = pick_one(population, fitness)
        order = cross_over(order1, order2, iters)

        if iters % 3 == 0:
            order = mutate(distance_matrix, order, P_SIZE, iters, m_rate)
        order_to_add = calculate_best(distance_matrix, [order, order3])[0]
        next_gen.append(order_to_add)  # adds the individual to the next generation
    return next_gen


def calculate_best(distance_matrix, population):
    # calculates the best individual in the population
    best = population[0]
    d = calculate_distance(distance_matrix, best)
    for i in range(1, len(population)):
        temp = calculate_distance(distance_matrix, population[i])
        if temp < d:
            d = temp
            best = population[i]
    return best, d


def natural_calamity(population, best, intensity):
    # function to simulate a natural calamity
    n = len(population)
    for i in range(intensity):
        j = np.random.randint(1, n)
        population[j] = best.copy()


distance_matrix = np.loadtxt(open("testCase50.csv", "rb"), delimiter=",").reshape(50, 50)

n = len(distance_matrix)  # number of cities
P_SIZE = 10  # size of the population
MAX_ITERS = 200
population = []
fitness = []
iter_no = []
tour_len = []
order = np.arange(n).tolist()
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(8, 5.5))
best_ever = None
optimal_distance = math.inf
iters = 0

start_time = time.time()

create_population(population, P_SIZE, order, fitness)
calculate_fitness(distance_matrix, population, fitness)
normalize_fitness(fitness, P_SIZE)

while iters < MAX_ITERS:
    if (iters + 1) % 37 == 0:
        # once in every 37 generations there is a natural calamity which whipes out certain phenotypes
        natural_calamity(population, best_ever, np.random.randint(1, math.ceil(0.5 + P_SIZE / 2)))
    population = next_generation(distance_matrix, population, P_SIZE, fitness, iters, best_ever)
    calculate_fitness(distance_matrix, population, fitness)
    normalize_fitness(fitness, P_SIZE)

    best, d_temp = calculate_best(distance_matrix, population)
    if d_temp < optimal_distance:
        optimal_distance = d_temp
        best_ever = best.copy()
        iter_no.append(iters)
        tour_len.append(optimal_distance)
        print(best_ever, optimal_distance)
        # if optimal_distance < 7000:
        #     break
    iters += 1

end_time = time.time()

plt.plot(iter_no, tour_len, '-bx')
plt.xlabel("Iterations")
plt.ylabel("Tour length")
plt.title("Sequential Convergence")
plt.tight_layout()
plt.savefig("convergence50_seq.png")
plt.clf()

print("Best order:", best_ever)
print("Optimal distance:", optimal_distance)
print("Time taken:", end_time - start_time, "seconds")
