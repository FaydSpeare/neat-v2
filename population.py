import time

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

        self.init_population()


    def init_population(self):
        model = self.organism_type(self.num_inputs, self.num_outputs)
        self.population = [model.replicate().fully_weight_mutated() for _ in range(self.population_size)]

    def calculate_fitness(self):
        [organism.calculate_fitness() for organism in self.population]


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


    def next(self):

        self.calculate_fitness()
        self.speciate()
        self.cull_species()
        self.share_fitness()

        # Remove stale species

        # Remove any species that don't get offspring

        # Populate Offspring

        self.generation += 1


if __name__ == '__main__':
    start = time.time()
    population = Population(2, 1, Organism, 100)
    for i in range(100):
        population.next()
    print(time.time() - start)
    print()