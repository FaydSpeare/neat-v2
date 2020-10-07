import math

from neat.species import Species


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
        self.best_ever_organism = None
        self.solvers = list()
        self.stale_count = 0
        self.init_population()


    def init_population(self):
        model = self.organism_type(self.num_inputs, self.num_outputs)
        self.population = [model.replicate().fully_weight_mutated() for _ in range(self.population_size)]


    def calculate_fitness(self):
        [organism.calculate_fitness() for organism in self.population]
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_generation_fitness = round(self.population[0].fitness, 4)
        if self.best_ever_fitness is None or self.best_generation_fitness > self.best_ever_fitness:
            self.best_ever_fitness = self.best_generation_fitness
            self.stale_count = 0
            self.best_ever_organism = self.population[0].replicate()
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

        # Remove all species aside from the top 2 if the whole population
        # has been stale for more than 20 generations
        if self.stale_count > 20:
            self.species = self.species[:2]

        total_population_fitness = sum([species.get_total_shared_fitness() for species in self.species])
        offspring = list()
        living_species = set()

        for species in self.species:

            # Assign number of offspring based on total fitness proportion
            total_species_fitness = species.get_total_shared_fitness()
            proportion = total_species_fitness / total_population_fitness
            num_offspring = math.floor(proportion * self.population_size)

            # Add champion if 5 or more offspring
            if num_offspring >= 5:
                offspring.append(species.get_champion().replicate())
                num_offspring -= 1

            # Add the remaining offspring
            offspring.extend(species.create_offspring(num_offspring))
            if num_offspring > 0: living_species.add(species)

        # TODO this is positively gross looking

        if len(offspring) < self.population_size:
            for species in self.species:
                total_species_fitness = species.get_total_shared_fitness()
                proportion = total_species_fitness / total_population_fitness
                num_offspring = math.ceil(proportion * self.population_size)
                num_offspring = min(self.population_size - len(offspring), num_offspring)
                offspring.extend(species.create_offspring(num_offspring))
                if num_offspring > 0: living_species.add(species)
                if len(offspring) == self.population_size: break

        self.species = list(living_species)
        self.population = offspring


    def eliminate_stale_species(self):

        non_stale_species = list()
        for species in self.species:

            # If a species has been stale for 15 generations eliminate it, unless
            # it is one of the top 2 species.
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


    def next(self, assessor_function):

        self.calculate_fitness()

        # Used to determine if any organisms can solve the task yet
        if assessor_function is not None:
            if self.assess_organisms(assessor_function):
                return True

        self.speciate()
        self.sort_species()
        self.cull_species()
        self.share_fitness()
        self.eliminate_stale_species()
        self.create_next_generation()
        self.generation += 1
        self.print_generation_info()


    def print_generation_info(self):
        print(("Gen: " + str(self.generation)).ljust(10), end="")
        print(("Species: " + str(len(self.species))).ljust(15), end="")
        print("Fitness: " + str(self.best_generation_fitness))


    def assess_organisms(self, assessor_function):
        for organism in self.population:
            if assessor_function(organism):
                self.solvers.append(organism.replicate())
        if self.solvers: return True


    def get_best_ever_organism(self):
        return self.best_ever_organism
