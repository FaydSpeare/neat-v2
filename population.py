import time
import math
import numpy as np

from species import Species
from organism import Organism


class Population:

    def __init__(self, num_inputs, num_outputs, organism, population_size):

        self.organism_type = organism
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.population_size = population_size
        self.generation = 0
        self.population = None
        self.species = list()
        self.best_generation_fitness = None
        self.best_ever_fitness = None
        self.champion = None
        self.stale_count = 0
        self.init_population()


    def init_population(self):
        model = self.organism_type(self.num_inputs, self.num_outputs)
        self.population = [model.replicate().fully_weight_mutated() for _ in range(self.population_size)]


    def calculate_fitness(self):
        [organism.calculate_fitness() for organism in self.population]
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_generation_fitness = self.population[0].fitness
        if self.best_ever_fitness is None or self.best_generation_fitness > self.best_ever_fitness:
            self.best_ever_fitness = self.best_generation_fitness
            self.stale_count = 0
            self.champion = self.population[0].replicate()
        else: self.stale_count += 1


    def share_fitness(self):
        [species.share_fitness() for species in self.species]


    def speciate(self):

        # Randomly assign representative genome
        [species.assign_genome() for species in self.species]

        # Clear species
        [species.clear() for species in self.species]

        # Assign organisms to species
        for organism in self.population:

            for species in self.species:

                if species.is_compatible_with(organism):
                    species.add_organism(organism)
                    break

            # Create new species
            else: self.species.append(Species(organism))

        # Remove any species without organisms
        self.species = [species for species in self.species if species.organisms]


    def cull_species(self):
        [species.eliminate_worst_organisms() for species in self.species]


    def create_next_generation(self):

        if self.stale_count > 20:
            self.species = self.species[:2]

        total_population_fitness = sum([species.get_total_shared_fitness() for species in self.species])
        total_offspring = 0
        offspring = list()

        living_species = set()

        for species in self.species:
            total_species_fitness = species.get_total_shared_fitness()
            proportion = total_species_fitness / total_population_fitness
            num_offspring = math.floor(proportion * self.population_size)
            total_offspring += num_offspring
            if num_offspring >= 5:
                offspring.append(species.get_champion().replicate())
                num_offspring -= 1
            offspring.extend(species.create_offspring(num_offspring))
            if num_offspring > 0: living_species.add(species)


        if total_offspring < self.population_size:
            for species in self.species:
                total_species_fitness = species.get_total_shared_fitness()
                proportion = total_species_fitness / total_population_fitness
                num_offspring = math.ceil(proportion * self.population_size)
                num_offspring = min(self.population_size - total_offspring, num_offspring)
                offspring.extend(species.create_offspring(num_offspring))
                if num_offspring > 0: living_species.add(species)
                total_offspring += num_offspring
                if total_offspring == self.population_size: break

        self.species = list(living_species)
        self.population = offspring


    def eliminate_stale_species(self):

        non_stale_species = list()
        for species in self.species:
            if species.stale_count < 15 or self.species.index(species) < 2:
                non_stale_species.append(species)
        self.species = non_stale_species

        if len(self.species) == 0: print("All species died.")


    def sort_species(self):

        for species in self.species:
            species.best_generation_fitness = max([organism.fitness for organism in species.organisms])
            if species.best_ever_fitness is None or species.best_generation_fitness > species.best_ever_fitness:
                species.best_ever_fitness = species.best_generation_fitness
                species.stale_count = 0
            else: species.stale_count += 1

        self.species.sort(key=lambda x: x.best_generation_fitness, reverse=True)


    def next(self):

        self.calculate_fitness()

        if (x := self.assess()) is not None:
            return x


        self.speciate()
        self.sort_species()
        self.cull_species()
        self.share_fitness()
        self.eliminate_stale_species()
        self.create_next_generation()
        self.generation += 1

        print(("Gen: " + str(self.generation)).ljust(10), \
              ("Species: " + str(len(self.species))).ljust(12), \
              ("Pop: " + str(len(self.population))).ljust(10), \
              self.best_generation_fitness)


    def assess(self):
        for organism in self.population:
            x = 0
            x += organism.think([0, 0])[0] < 0.5
            x += organism.think([0, 1])[0] > 0.5
            x += organism.think([1, 0])[0] > 0.5
            x += organism.think([1, 1])[0] < 0.5
            if x == 4: return organism
        return None


    def get_champion(self):
        return self.champion


if __name__ == '__main__':
    start = time.time()


    its = list()
    hids = list()
    N = 50
    for i in range(N):
        population = Population(2, 1, Organism, 100)
        it = 0
        while True:
            it += 1
            if (x := population.next()) is not None:
                its.append(it)
                hids.append(len(x.brain.hidden_nodes))
                champ = population.get_champion()
                break
    print("Its", np.mean(its), np.std(its))
    print("Hiddens", np.mean(hids), np.std(hids))

    print(time.time() - start)
    print()