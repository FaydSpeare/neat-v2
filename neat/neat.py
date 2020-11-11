from neat.population import Population
from neat.organism import Organism

class Neat:

    def __init__(self, organism_type, config={}, assessor_function=None, verbose=True):
        self.organism_type = organism_type
        self.check_organism_type()
        self.verbose = verbose
        self.assessor_function = assessor_function
        self.config = dict(self.default_config(), **config)
        self.population = Population(organism_type, self.config)


    def run(self, generations=None):
        while True:
            if self.population.next(self.assessor_function):
                print("\nSolver Organism/s found.\n")
                return
            if generations is not None and self.population.generation > generations:
                break


    def get_solvers(self):
        return self.population.solvers


    def check_organism_type(self):
        if not issubclass(self.organism_type, Organism):
            raise Exception("organism_type must be a subclass of Organism")
        if not callable(getattr(self.organism_type, 'calculate_fitness', None)):
            raise Exception("organism_type must implement the calculate_fitness function")

    def default_config(self):

        return {

            'population_size' :      100,

            'num_inputs'  :            1,
            'num_outputs' :            1,

            'add_node_prob' :       0.01,
            'add_conn_prob' :        0.1,
            'mutate_weight_prob' :   0.8,
            'replace_weight_prob' :  0.1,

            'weight_init' : 'gauss',
            'weight_init_params' : [0, 1],
            'weight_mutate_step' :   0.01,

            'activations' : ['sigmoid'],
            'activation_weights' : [1],
            'output_activation' : 'sigmoid',

            'weight_min' : -3,
            'weight_max' : 3,

            'compat_disjoint_coeff' : 1.0,
            'compat_weight_coeff' : 3.0,
            'compat_threshold' : 4.0,

            'survival_threshold': 0.2,
            'champion' : True,
            'min_species_size' : 2,
            'elitists' : 2,

            'custom_print_fields' : [],
            'stats_file' : None,

            'dup_parent' : 0.25,

            'population_stag' : 20,
            'species_stag' : 15

        }





