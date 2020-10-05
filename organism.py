import random

from network import Network

class Organism:

    def __init__(self, num_inputs, num_outputs, replication=False):

        self.brain = None
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fitness = None
        self.adjusted_fitness = None

        if not replication:
            self.brain = Network(num_inputs, num_inputs)

    def think(self, sensors):
        return self.brain.forward(sensors)

    def replicate(self):
        copy = self.__class__(self.num_inputs, self.num_outputs, replication=True)
        copy.brain = self.brain.replicate()
        return copy

    def fully_weight_mutated(self):
        self.brain.full_weight_mutation()
        return self

    def reset(self):
        self.fitness = None
        self.adjusted_fitness = None

    def __lt__(self, other):
        return self.fitness < other.fitness


    def calculate_fitness(self):
        self.fitness = random.random()
