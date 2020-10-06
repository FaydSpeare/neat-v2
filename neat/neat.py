from neat.population import Population
from neat.organism import Organism

class Neat:

    def __init__(self, num_inputs, num_outputs, organism_type, population_size=100, assessor_function=None, verbose=True):
        self.organism_type = organism_type
        self.check_organism_type()
        self.verbose = verbose
        self.assessor_function = assessor_function
        self.population = Population(num_inputs, num_outputs, organism_type, population_size)


    def run(self, generations=100):
        for i in range(generations):
            if self.population.next(self.assessor_function):
                print("\nSolver Organism/s found.\n")
                return


    def get_solvers(self):
        return self.population.solvers


    def check_organism_type(self):
        if not issubclass(self.organism_type, Organism):
            raise Exception("organism_type must be a subclass of Organism")
        if not callable(getattr(self.organism_type, 'calculate_fitness', None)):
            raise Exception("organism_type must implement the calculate_fitness function")



