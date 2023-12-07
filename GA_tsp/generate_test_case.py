import random

import numpy as np

n = int(input("Enter number of cities\n>>>"))
cities = [[0.0] * n for _ in range(n)]  # Tạo ma trận nxn với các phần tử ban đầu là 0

for i in range(n):
    for j in range(n):
        if i == j:
            cities[i][j] = 0.0
        else:
            cities[i][j] = random.uniform(20, 1000)

print(cities)
cities = np.array(cities)
cities.tofile('testCase100.csv', sep=',')
