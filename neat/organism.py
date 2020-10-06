import random

from neat.network import Network

class Organism:

    def __init__(self, num_inputs, num_outputs, replication=False):

        self.brain = None
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fitness = None
        self.adjusted_fitness = None

        if not replication:
            self.brain = Network(num_inputs, num_outputs)


    def think(self, sensors):
        self.brain.reset()
        return self.brain.forward(sensors)


    def replicate(self):
        copy = self.__class__(self.num_inputs, self.num_outputs, replication=True)
        copy.brain = self.brain.replicate()
        copy.fitness = self.fitness
        return copy


    def fully_weight_mutated(self):
        self.brain.full_weight_mutation()
        return self


    def mutate(self):
        self.brain.mutate()


    def reset(self):
        self.fitness = None
        self.adjusted_fitness = None


    def crossover(self, other):
        offspring = self.replicate()

        for conn in offspring.brain.connections:
            for other_conn in other.brain.connections:
                if conn.number == other_conn.number:
                    if random.random() < 0.5: conn.weight = other_conn.weight

                    if not conn.enabled or not other_conn.enabled:
                        if random.random() < 0.75: conn.enabled = False
                        else: conn.enabled = True
                else:
                    if not conn.enabled:
                        if random.random() < 0.25: conn.enabled = True

        for conn in offspring.brain.bias_connections:
            for other_conn in other.brain.bias_connections:
                if conn.number == other_conn.number and random.random() < 0.5:
                    conn.weight = other_conn.weight



        return offspring


    def __lt__(self, other):
        return self.fitness > other.fitness

