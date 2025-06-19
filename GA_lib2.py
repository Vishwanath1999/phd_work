import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy.io import savemat

plt.style.use('physrev.mplstyle')

class geneticalgorithm:
    def __init__(self,
                 function,
                 dimension,
                 variable_type='real',
                 variable_boundaries=None,
                 variable_type_mixed=None,
                 function_timeout=10,
                 algorithm_parameters=None,
                 convergence_curve=True,
                 progress_bar=True,
                 parallel=False,
                 n_processes=None,
                 random_seed=None,
                 fitness_threshold=None):
        """
        Initializes the genetic algorithm with the given parameters.
        """
        self.funct = function
        self.dim = int(dimension)
        self.var_type = variable_type
        self.var_bound = np.array(variable_boundaries)
        self.var_type_mixed = variable_type_mixed
        self.function_timeout = function_timeout
        self.convergence_curve = convergence_curve
        self.progress_bar = progress_bar
        self.parallel = parallel
        self.n_processes = n_processes or mp.cpu_count()
        self.fitness_threshold = fitness_threshold

        # Show max and used processes
        max_procs = os.cpu_count()
        if self.parallel:
            print(f"[GA] Max available processes: {max_procs}")
            print(f"[GA] Processes used for parallel evaluation: {self.n_processes}")

        if random_seed is not None:
            np.random.seed(random_seed)

        # Default algorithm parameters
        default_params = {
            'max_num_iteration': 1000,
            'population_size': 100,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }
        if algorithm_parameters is not None:
            default_params.update(algorithm_parameters)
        self.param = default_params

        self.population = np.zeros((self.param['population_size'], self.dim + 1))
        self.best_function = []
        self.best_variable = []

    def run(self, save_curve_path=None):
        self.__initialize_population()
        self.__evaluate_population(self.population)

        count = 0
        best_function = self.population[0, -1]
        best_variable = self.population[0, :-1].copy()
        curve = []
        self.all_fitness_per_gen = []

        gen_iter = range(self.param['max_num_iteration'])
        if self.progress_bar:
            gen_iter = tqdm(gen_iter, desc='Generations', ncols=60)

        for iteration in gen_iter:
            # Sort by fitness (lower is better)
            self.population = self.population[self.population[:, -1].argsort()]
            if self.population[0, -1] < best_function:
                best_function = self.population[0, -1]
                best_variable = self.population[0, :-1].copy()
                count = 0
            else:
                count += 1

            self.all_fitness_per_gen.append(self.population[:, -1].copy())
            curve.append(best_function)

            # Early stopping if fitness threshold is reached
            if self.fitness_threshold is not None and best_function <= self.fitness_threshold:
                if self.progress_bar:
                    print(f"Early stopping: fitness threshold {self.fitness_threshold} reached.")
                break

            # Elitism (retain fitness for elites)
            n_elit = int(np.ceil(self.param['elit_ratio'] * self.param['population_size']))
            elit_pop = self.population[:n_elit, :].copy()

            # Parents selection
            n_parents = int(np.ceil(self.param['parents_portion'] * self.param['population_size']))
            parents = self.population[:n_parents, :].copy()

            # Crossover
            n_children = self.param['population_size'] - n_elit
            children = np.zeros((n_children, self.dim))
            for i in range(n_children):
                parent1 = parents[np.random.randint(0, n_parents), :-1]
                parent2 = parents[np.random.randint(0, n_parents), :-1]
                children[i] = self.__crossover(parent1, parent2)

            # Vectorized mutation
            self.__mutate_population(children)

            # Create new population (retain fitness for elites)
            new_population = np.vstack((elit_pop[:, :-1], children))
            new_population = np.hstack((new_population, np.zeros((new_population.shape[0], 1))))
            self.__evaluate_population(new_population)
            self.population = new_population

            if self.param['max_iteration_without_improv'] is not None:
                if count > self.param['max_iteration_without_improv']:
                    if self.progress_bar:
                        print("Stopping: no improvement for max_iteration_without_improv generations.")
                    break

        if self.progress_bar:
            print()

        self.fitness_curve = curve

        if self.convergence_curve:
            self.plot_fitness_curve(save_path=save_curve_path)
            self.plot_population_fitness(save_path=save_curve_path)
            # Save all generations' fitness to a .mat file
            save_path_mat = save_curve_path + '/population_fitness.mat' if save_curve_path else 'population_fitness.mat'
            savemat(save_path_mat, {'fitness': np.array(self.all_fitness_per_gen)})

        self.best_function = best_function
        self.best_variable = best_variable
        return best_variable, best_function

    def plot_population_fitness(self, save_path=None):
        """
        Plot the fitness of the entire population per generation,
        highlighting the best solution in each generation.
        """
        if not self.all_fitness_per_gen:
            print("No population fitness data to plot.")
            return
        fitness_matrix = np.array(self.all_fitness_per_gen)
        generations = np.arange(fitness_matrix.shape[0])
        plt.figure(figsize=(10, 6))
        plt.plot(generations, fitness_matrix, '*', color='black', markersize=3, alpha=0.7, label='Population')
        best_fitness = np.min(fitness_matrix, axis=1)
        plt.plot(generations, best_fitness, 'ro-', linewidth=2, label='Best Fitness', markersize=3)
        plt.xlabel('Generation', fontsize=16)
        plt.ylabel('Fitness', fontsize=16)
        plt.title('Population Fitness per Generation', fontsize=18, fontweight='bold')
        plt.legend()
        plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        fig_name = 'population_fitness_curve.png'
        if save_path is not None:
            save_path = save_path if save_path.endswith('.png') else f"{save_path}/{fig_name}"
        else:
            save_path = fig_name
        print(f"Saving population fitness plot to {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_fitness_curve(self, save_path=None):
        """
        Plot and optionally save the fitness (best function value) vs generations.
        """
        if not self.fitness_curve:
            print("No fitness curve data to plot.")
            return
        plt.figure(figsize=(8, 5))
        plt.plot(self.fitness_curve, label='Best Fitness')
        plt.xlabel('Generation', fontsize=16)
        plt.ylabel('Best Fitness', fontsize=16)
        plt.title('Fitness Curve', fontsize=18, fontweight='bold')
        plt.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        fig_name = 'fitness_curve.png'
        if save_path is not None:
            save_path = save_path if save_path.endswith('.png') else f"{save_path}/{fig_name}"
        else:
            save_path = fig_name
        print(f"Saving fitness curve plot to {save_path}")
        plt.savefig(save_path, dpi=300)
        plt.show()

    def __initialize_population(self):
        for i in range(self.param['population_size']):
            self.population[i, :-1] = self.__generate_individual()

    def __generate_individual(self):
        individual = np.zeros(self.dim)
        for j in range(self.dim):
            if self.var_type == 'real':
                individual[j] = np.random.uniform(self.var_bound[j, 0], self.var_bound[j, 1])
            elif self.var_type == 'int':
                individual[j] = np.random.randint(self.var_bound[j, 0], self.var_bound[j, 1] + 1)
            elif self.var_type == 'binary':
                individual[j] = np.random.randint(0, 2)
            else:
                raise ValueError("Unsupported variable type.")
        return individual

    def __evaluate_population(self, population):
        if self.parallel:
            with mp.Pool(processes=self.n_processes) as pool:
                results = []
                for res in tqdm(pool.imap(self.funct, [ind[:-1] for ind in population]),
                                total=len(population),
                                desc="Fitness (parallel)",
                                disable=not self.progress_bar,
                                leave=False, ncols=100):
                    results.append(res)
                for i in range(len(population)):
                    population[i, -1] = results[i]
        else:
            for i, ind in enumerate(tqdm(population, desc="Fitness (serial)",
                                         disable=not self.progress_bar, leave=False, ncols=100)):
                population[i, -1] = self.funct(ind[:-1])

    def __crossover(self, parent1, parent2):
        if self.param['crossover_type'] == 'uniform':
            mask = np.random.rand(self.dim) < 0.5
            child = np.where(mask, parent1, parent2)
        else:  # 'single_point'
            point = np.random.randint(1, self.dim)
            child = np.concatenate([parent1[:point], parent2[point:]])
        return child

    def __mutate_population(self, children):
        """
        Vectorized mutation for the children array.
        """
        mutation_mask = np.random.rand(*children.shape) < self.param['mutation_probability']
        for j in range(self.dim):
            if self.var_type == 'real':
                random_values = np.random.uniform(self.var_bound[j, 0], self.var_bound[j, 1], size=children.shape[0])
            elif self.var_type == 'int':
                random_values = np.random.randint(self.var_bound[j, 0], self.var_bound[j, 1] + 1, size=children.shape[0])
            elif self.var_type == 'binary':
                random_values = np.random.randint(0, 2, size=children.shape[0])
            else:
                raise ValueError("Unsupported variable type.")
            children[:, j] = np.where(mutation_mask[:, j], random_values, children[:, j])