import math
import random

import numpy as np
from mpi4py import MPI

from flask import Flask, request, jsonify
from flask_cors import CORS
from json import JSONEncoder

def calculate_distance(distance_matrix, order):
    # calculates the distance between the cities in the order given
    distance = 0
    for i in range(len(order) - 1):
        distance += distance_matrix[order[i]][order[i + 1]]

    return distance


def swap(arr, i, j):
    # swaps the cities at two locations in our tour
    arr[i], arr[j] = arr[j], arr[i]


def create_population(population, P_SIZE, order, fitness):
    # create a population of random orders
    for i in range(P_SIZE):
        new_order = order.copy()
        shuffle_array(new_order)
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
    random.shuffle(population)
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
    normalize_order(order)
    if iters % 12 == 0 and mutation_rate != 0:  # If the current iteration is divisible by 12 and the mutation rate is not zero, the function shuffles the entire order randomly.
        shuffle_array(order)
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
                    normalize_order(order)
                    temp = calculate_distance(distance_matrix, order)  # calculates the distance of the new order
                    if temp < d:  # if the new order is better than the previous order, the new order is retained
                        d = temp  # updates the distance
                        if iters < 100 + int(1000 / P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return order
                    else:
                        swap(order, k, j)
                        normalize_order(order)
            else:  # if the order is not reversed
                for j in range(math.ceil(n / 2)):
                    k = (i + j) % n
                    swap(order, k, j)
                    normalize_order(order)
                    temp = calculate_distance(distance_matrix, order)
                    if temp < d:
                        d = temp
                        if iters < 100 + int(1000 / P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return order
                    else:
                        swap(order, k, j)
                        normalize_order(order)

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
                    normalize_order(order)
                    temp = calculate_distance(distance_matrix, order)
                    if temp < d:
                        d = temp
                        if iters < 100 + int(1000 / P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return order
                    else:
                        order[i:k + 1] = np.flip(order[i:k + 1])
                        normalize_order(order)
            else:
                for j in range(int(n / 2)):
                    k = (i + j) % n
                    if k < i:
                        temp = i
                        i = k
                        k = temp
                    order[i:k + 1] = np.flip(order[i:k + 1])
                    normalize_order(order)
                    temp = calculate_distance(distance_matrix, order)
                    if temp < d:
                        d = temp
                        if iters < 100 + int(1000 / P_SIZE) and np.random.randint(0, 100) % 17 == 0:
                            return order
                    else:
                        order[i:k + 1] = np.flip(order[i:k + 1])
                        normalize_order(order)
    return order


def cross_over(order1, order2, iters):
    # function to perform cross over between two orders
    n = len(order1)  # number of cities
    i1 = np.random.randint(1, n - 1)  # selects a random city
    i2 = np.random.randint(1, n - 1)  # selects another random city

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
    order[n - 1] = 0
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


def normalize_order(order):
    # normalize the order
    while order.count(0) > 0:
        order.remove(0)
    order.append(0)
    order.insert(0, 0)


def shuffle_array(arr):
    # shuffle the array
    np.random.shuffle(arr)
    normalize_order(arr)




matrix = [[0, 9, 24, 13, 4, 9, 3],
          [9, 0, 17, 6, 1, 2, 3],
          [24, 17, 0, 11, 8, 9, 7],
          [13, 6, 11, 0, 5, 9, 3],
          [4, 1, 8, 5, 0, 9, 3],
          [9, 2, 9, 9, 9, 0, 3],
          [3, 3, 3, 3, 3, 3, 0]]


def genetic_algorithm(distance_matrix):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    is_root = rank == 0

    if is_root:
        n = len(distance_matrix)  # number of cities
        P_SIZE = 3  # size of the population
        MAX_ITERS = 10
        data = (n, P_SIZE, MAX_ITERS, distance_matrix)
        for i in range(1, size):
            comm.send(data, dest=i, tag=i)
    else:
        n, P_SIZE, MAX_ITERS, distance_matrix = comm.recv(source=0, tag=rank)

    population = []
    fitness = []
    order = np.arange(n + 1).tolist()
    order[len(order) - 1] = 0

    best_ever = None
    optimal_distance = math.inf
    iters = 0
    create_population(population, P_SIZE, order, fitness)
    calculate_fitness(distance_matrix, population, fitness)
    normalize_fitness(fitness, P_SIZE)

    while iters < MAX_ITERS:
        if (iters + 1) % 5 == 0 and size > 1:
            # once in 5 generations there is migration of individuals between islands
            random.shuffle(population)
            comm.send(population[0:P_SIZE // 2], dest=(rank + 1) % size, tag=rank)
            temp = comm.recv(source=(rank - 1) % size, tag=(rank - 1) % size)
            population[0:P_SIZE // 2] = temp
        if (iters + 1) % 37 == 0:
            # once in every 37 generations there is a natural calamity which whipes out certain phenotypes
            natural_calamity(population, best_ever, np.random.randint(1, math.ceil(0.5 + P_SIZE / 2)))
        population = next_generation(distance_matrix, population, P_SIZE, fitness, iters, best_ever)
        calculate_fitness(distance_matrix, population, fitness)
        normalize_fitness(fitness, P_SIZE)

        best_ever, d_temp = calculate_best(distance_matrix, population)
        temp_pop = comm.gather(best_ever, root=0)

        if is_root:
            best_in_the_world, d = calculate_best(distance_matrix, temp_pop)
            if d < optimal_distance:
                optimal_distance = d
                best_ever = best_in_the_world
        iters += 1
    if is_root:
        return {"best_ever":best_ever, "optimal_dist": optimal_distance}


class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)  # Chuyển đổi thành kiểu int trước khi chuyển thành JSON
        return super().default(obj)

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder

CORS(app, origins='http://localhost:3000')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Ví dụ: Tăng giới hạn thành 16MB


@app.route("/matrix", methods = ['POST', 'GET'])
def recieve_matrix():
    data = request.get_json()
    print(data)
    result = genetic_algorithm(data['distanceMatrix'])
    print(result)
    result['best_ever'] = [int(x) for x in result['best_ever']]
    result['optimal_dist'] = float(result['optimal_dist'])
    return jsonify(result)

# //result = genetic_algorithm(data)
# best_ever = result[0]
# optimal_dist = result[1]

# @app.route("/")
# def data():
#     data = {
#         "best_ever": best_ever.tolist(),
#         "optimal_dist":optimal_dist
#     }
#     return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
    
