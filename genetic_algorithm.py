import numpy as np
from network import NeuralNetwork
from utils import decode, nguyen_widrow_init


class StructuralGeneticAlgorithm:
    def __init__(self, X, Y, Z_real, pop_size=20, mutation_rate=0.1,
                 max_neurons=20, max_error=0.01, max_layers=5):
        self.X, self.Y, self.Z_real = X, Y, Z_real
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.max_neurons = max_neurons
        self.max_error = max_error
        self.max_layers = max_layers

        self.population = [self.random_chromosome() for _ in range(pop_size)]

    def random_structure(self):
        layers = np.random.randint(1, self.max_layers + 1)
        return [2] + [np.random.randint(2, self.max_neurons) for _ in range(layers)] + [1]

    def random_chromosome(self):
        structure = self.random_structure()
        weights = nguyen_widrow_init(structure)
        return (structure, weights)

    def fitness(self, chromosome):
        structure, genome = chromosome
        nn = NeuralNetwork(structure)
        decode(nn, genome)

        inputs = np.vstack((self.X.ravel(), self.Y.ravel()))
        Z_pred = nn.forward(inputs).reshape(self.X.shape)
        return np.mean((self.Z_real - Z_pred) ** 2)

    def select_parents(self, scores):
        probs = (1 / (1 + np.array(scores)))
        probs = probs / probs.sum()
        idx = np.random.choice(len(scores), size=2, p=probs)
        return self.population[idx[0]], self.population[idx[1]]

    def crossover(self, p1, p2):
        s1, g1 = p1
        s2, g2 = p2

        point = min(len(s1), len(s2)) // 2
        child_structure = s1[:point] + s2[point:]

        child_weights = nguyen_widrow_init(child_structure)
        return (child_structure, child_weights)

    def mutate(self, chromosome):
        structure, genome = chromosome

        if np.random.rand() < self.mutation_rate:
            idx = np.random.randint(1, len(structure) - 1)
            structure[idx] = np.random.randint(2, self.max_neurons)

            genome = nguyen_widrow_init(structure)

        for i in range(len(genome)):
            if np.random.rand() < self.mutation_rate:
                genome[i] += np.random.normal(0, 0.2)

        return (structure, genome)

    def evolve(self, generations, info_callback=None):
        best_history = []

        for gen in range(generations):
            scores = [self.fitness(c) for c in self.population]
            best_idx = np.argmin(scores)
            best = self.population[best_idx]

            best_history.append((gen, best[0], scores[best_idx]))

            if info_callback:
                info_callback(gen, best[0], scores[best_idx])

            if scores[best_idx] <= self.max_error:
                return best, best_history

            new_pop = [best]

            while len(new_pop) < self.pop_size:
                p1, p2 = self.select_parents(scores)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)

            self.population = new_pop

        scores = [self.fitness(c) for c in self.population]
        best_idx = np.argmin(scores)
        return self.population[best_idx], best_history
