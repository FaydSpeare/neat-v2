import random
import itertools
import bisect
import numpy as np

species_counter = itertools.count(0)

class Species:

    def __init__(self, organism):

        self.number = next(species_counter)
        self.genome = organism.brain.replicate()
        self.organisms = [organism]

        self.best_ever_fitness = None
        self.best_generation_fitness = None
        self.stale_counter = 0


    def add_organism(self, organism):
        bisect.insort(self.organisms, organism)


    def clear(self):
        self.organisms.clear()
        self.best_generation_fitness = None


    def assign_genome(self):
        self.genome = random.choice(self.organisms).brain.replicate()


    def share_fitness(self):
        for organism in self.organisms:
            organism.adjusted_fitness = organism.fitness / len(self.organisms)


    def get_total_shared_fitness(self):
        return sum([organism.adjusted_fitness for organism in self.organisms])


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
        c1, c2, c3 = 1., 1., 3.
        threshold = 4.
        return (c1 * disjoint / N + c2 * excess / N + c3 * avg_weight_diff) < threshold


    def eliminate_worst_organisms(self):
        if len(self.organisms) > 2:
            remove_amount = len(self.organisms) // 2
            self.organisms = self.organisms[:-remove_amount]


    def parent_selector(self):

        fitness_list = np.array([organism.fitness for organism in self.organisms])
        portions = fitness_list / sum(fitness_list)
        cum_sum = np.cumsum(portions)

        while True:

            # Returns parents with probability proportional to their fitness
            parent_idx = bisect.bisect(cum_sum, random.random())
            yield self.organisms[parent_idx]


    def get_champion(self):
        return self.organisms[0]


    def create_offspring(self, amount):

        offspring = list()
        parent = self.parent_selector()

        for _ in range(amount):

            # Simply copy parent 25% of the time
            if random.random() < 0.25:
                offspring.append(next(parent).replicate())

            # Crossover parents
            else:
                strong_parent = next(parent)
                weak_parent = next(parent)
                if strong_parent.fitness < weak_parent.fitness:
                    strong_parent, weak_parent = weak_parent, strong_parent
                offspring.append(strong_parent.crossover(weak_parent))

        # Mutate each of the offspring
        [child.mutate() for child in offspring]

        return offspring

    # TODO structural mutations are currently being tracked in perpetuity. is think ok?
    # TODO it significantly reduces innovation number counts
    # reset_structure_tracker()