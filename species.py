import random
import itertools
import bisect

species_counter = itertools.count(0)

class Species:

    def __init__(self, organism):

        self.number = next(species_counter)
        self.genome = organism.brain.replicate()
        self.organisms = [organism]

    def add_organism(self, organism):
        bisect.insort(self.organisms, organism)

    def clear(self):
        self.organisms.clear()

    def assign_genome(self):
        self.genome = random.choice(self.organisms).brain.replicate()

    def share_fitness(self):
        for organism in self.organisms: organism.adjusted_fitness = organism.fitness / len(self.organisms)

    def is_compatible_with(self, organism):

        net1 = self.genome
        net2 = organism.brain

        conn_set1 = set([conn.number for conn in net1.connections])
        conn_set2 = set([conn.number for conn in net2.connections])
        matching = conn_set1.intersection(conn_set2)
        excess_cutoff = min(max(conn_set1), max(conn_set2))
        full_set = conn_set1.union(conn_set2)
        excess = sum([conn > excess_cutoff for conn in full_set])
        disjoint = len(full_set) - excess - len(matching)

        innov_to_weight1 = {conn.number : conn.weight for conn in net1.connections if conn.number in matching}
        innov_to_weight2 = {conn.number: conn.weight for conn in net2.connections if conn.number in matching}
        avg_weight_diff = sum(abs(innov_to_weight1[i] - innov_to_weight2[i]) for i in matching) / len(matching)

        N = max(len(conn_set1), len(conn_set2))
        if N < 20: N = 1.

        return (1. * disjoint / N + 1. * excess / N + 3. * avg_weight_diff) < 3.0


    def eliminate_worst_organisms(self):
        if len(self.organisms) > 2:
            remove_amount = len(self.organisms) // 2
            self.organisms = self.organisms[:-remove_amount]


