import numpy as np
from utils import decode, nguyen_widrow_init
from network import NeuralNetwork

class GeneticAlgorithm:
    def __init__(self, nn_shape, X, Y, Z_real, pop_size=30, mutation_rate=0.1):
        self.nn_shape = nn_shape
        self.X, self.Y, self.Z_real = X, Y, Z_real
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate

        self.param_count = sum(
            nn_shape[i] * nn_shape[i + 1] + nn_shape[i + 1]
            for i in range(len(nn_shape) - 1)
        )

        self.population = [
            nguyen_widrow_init(nn_shape) if i < 5 else np.random.uniform(-1, 1, self.param_count)
            for i in range(pop_size)
        ]

    def fitness(self, genome):
        nn = NeuralNetwork(self.nn_shape)
        decode(nn, genome)
        inputs = np.vstack((self.X.ravel(), self.Y.ravel()))
        Z_pred = nn.forward(inputs).reshape(self.X.shape)
        return np.mean((self.Z_real - Z_pred) ** 2)

    def select_parents(self, scores):
        probs = (1 / (1 + np.array(scores)))
        probs /= probs.sum()
        idx = np.random.choice(range(self.pop_size), size=2, p=probs)
        return self.population[idx[0]], self.population[idx[1]]

    def crossover(self, p1, p2):
        point = np.random.randint(0, len(p1))
        return np.concatenate((p1[:point], p2[point:]))

    def mutate(self, genome):
        for i in range(len(genome)):
            if np.random.rand() < self.mutation_rate:
                genome[i] += np.random.normal(0, 0.3)
        return genome

    def evolve(self, generations=30):
        best_errors = []
        for gen in range(generations):
            scores = [self.fitness(g) for g in self.population]
            best_idx = np.argmin(scores)
            best_errors.append(scores[best_idx])
            print(f"Покоління {gen}: найкраща похибка = {scores[best_idx]:.5f}")
            new_pop = [self.population[best_idx].copy()]  # елітизм
            while len(new_pop) < self.pop_size:
                p1, p2 = self.select_parents(scores)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)
            self.population = new_pop
        print(f"Завершено\n")
        return self.population[best_idx], best_errors
