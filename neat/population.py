import math

from neat.species import Species


class Population:

    def __init__(self, organism, config):

        self.organism_type = organism
        self.config = config
        self.num_inputs = config['num_inputs']
        self.num_outputs = config['num_outputs']
        self.population_size = config['population_size']
        self.generation = 0
        self.population = None
        self.species = list()
        self.best_generation_fitness = None
        self.best_ever_fitness = None
        self.best_ever_organism = None
        self.best_generation_organism = None
        self.solvers = list()
        self.stale_count = 0
        self.init_population()

        self.file_stats = {
            'generation' : [],
            'fitness' : [],
            'species' : []
        }

        for attr in self.config['custom_print_fields']:
            self.file_stats[attr] = list()


    def init_population(self):
        model = self.organism_type(self.config)
        self.population = [model.replicate().fully_weight_mutated() for _ in range(self.population_size)]


    def calculate_fitness(self):
        [organism.calculate_fitness() for organism in self.population]
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_generation_fitness = round(self.population[0].fitness, 4)
        self.best_generation_organism = self.population[0]
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
            else: self.species.append(Species(self.config, organism))

        # Remove any species without organisms
        self.species = [species for species in self.species if species.organisms]


    def cull_species(self):
        [species.eliminate_worst_organisms() for species in self.species]

        # Remove eliminated organisms from population
        new_population = list()
        [new_population.extend(species.organisms) for species in self.species]
        self.population = new_population



    def create_next_generation(self):

        offspring = list()
        living_species = set()

        # Remove all species aside from the top 2 if the whole population
        # has been stale for more than 20 generations
        if self.stale_count > self.config['population_stag']:
            self.species = self.species[:2]

        if self.config['champion']:
            offspring.append(self.best_generation_organism.replicate())

        fitness_list = [organism.adjusted_fitness for organism in self.population]
        min_fitness = min(fitness_list)
        max_fitness = max(fitness_list)
        fitness_range = max(1., max_fitness - min_fitness)

        adjusted_fitness_list = [(s.get_mean_shared_fitness() - min_fitness) / fitness_range for s in self.species]
        total_adjusted_fitness = sum(adjusted_fitness_list)

        #num_offspring_list = [math.floor(self.population_size * (((species.get_mean_shared_fitness() - min_fitness) / fitness_range) / total_adjusted_fitness)) for species in self.species]
        #print(adjusted_fitness_list)
        #print(num_offspring_list)

        for species in self.species:


            # Assign number of offspring based on total fitness proportion
            adjusted_fitness = (species.get_mean_shared_fitness() - min_fitness) / fitness_range
            proportion = 1. if total_adjusted_fitness == 0 else adjusted_fitness / total_adjusted_fitness
            num_offspring = math.floor(proportion * self.population_size)

            # Add champion if 5 or more offspring
            # TODO above or below?
            # Add elitists

            elitists = min(num_offspring, self.config['elitists'])
            elitists = min(elitists, len(species.organisms))
            for i in range(elitists):
                offspring.append(species.organisms[i])
            num_offspring -= elitists

            # Add the remaining offspring
            offspring.extend(species.create_offspring(num_offspring))
            if num_offspring > 0: living_species.add(species)

        # TODO this is positively gross looking

        if len(offspring) < self.population_size:
            for species in self.species:
                adjusted_fitness = (species.get_mean_shared_fitness() - min_fitness) / fitness_range
                proportion = adjusted_fitness / total_adjusted_fitness
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
            if species.stale_count < self.config['species_stag'] or self.species.index(species) < 2:
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
                if self.config['stats_file'] is not None: self.print_file()
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
        print(("Fitness: " + str(self.best_generation_fitness)).ljust(25), end="")
        print(("Size: " + str(len(self.population))).ljust(15), end="")

        if len(self.config['custom_print_fields']) > 0:
            custom_attrs = dict()
            for attr in self.config['custom_print_fields']:
                custom_attrs[attr] = getattr(self.best_generation_organism, attr)
            print("Custom_Attrs:", custom_attrs, end="")

        print()
        self.append_file_info()

    def append_file_info(self):

        self.file_stats['generation'].append(self.generation)
        self.file_stats['fitness'].append(self.best_generation_fitness)
        self.file_stats['species'].append(len(self.species))

        for attr in self.config['custom_print_fields']:
            self.file_stats[attr].append(getattr(self.best_generation_organism, attr))

    def print_file(self):

        f = open(self.config['stats_file'], "w")
        for key in self.file_stats:
            f.write(key)
            f.write(str(self.file_stats[key]))
            f.write('\n')
        f.close()



    def assess_organisms(self, assessor_function):
        for organism in self.population:
            if assessor_function(organism):
                self.solvers.append(organism)
        if self.solvers: return True


    def get_best_ever_organism(self):
        return self.best_ever_organism
