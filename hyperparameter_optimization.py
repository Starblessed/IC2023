from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse

import random
import statistics as stt
import numpy as np
import math

import cv2
from matplotlib import pyplot as plt

class Individual:
    def __init__(self, hparams: list, hparam_names: list, type: str, model=None):
        self.hparams = hparams
        self.hparam_names = hparam_names
        self.model = model
        self.score = 0
        self.type = type
        self.color = None
    def __repr__(self):
        hparam_str = ''
        for hparam, hparam_name in zip(self.hparams, self.hparam_names):
            hparam_str += f'{hparam_name}: {hparam}\n'
        return f'Individual of Type "{type(self)}"\nHParams:\n{hparam_str}'


class DTR_Individual(Individual):
    def __init__(self, hparams: list, hparam_names: list, model=None):
        super().__init__(hparams, hparam_names, 'DTR', model)
        self.color = f'#{format(int((hparams[0] - 1) * 2), "x")}ff{format(int((hparams[1] - 1) * 2), "x")}'


class Population:
    def __init__(self, type: str, individuals=[], hparam_names=[], generation=0):
        self.individuals = individuals
        self.generation = generation
        self.hparam_names = hparam_names
    
    def add_individual(self, individual: Individual):
        self.individuals.append(individual)

    def show_stats(self):
        pass

    def rank(self, mode: int):
        match mode:
            case 0:
                invert = False
            case 1:
                invert = True
            case _:
                raise ValueError(f"Invalid optimizing mode {mode}. Use 0 for minimize and 1 for maximize.")
        
        self.individuals = sorted(self.individuals, key=lambda x: x.score, reverse=invert)
    
    def update(self, new_population= list):
        self.individuals = new_population
        self.generation += 1


class DTR_GA:
    def __init__(self, metric: str, goal: int):
        self.hparam_names = ['max_depth', 'min_samples_split']
        self.population = Population('DTR')
        self.metric = metric
        self.goal = goal
        self.metric_history = []

    def create_individual(self, parents=[], val_range=[2, 8]):
        if parents == []:
            max_depth = random.randint(val_range[0], val_range[1])
            min_samples_split = random.randint(val_range[0], val_range[1])
        else:
            mother, father = parents
            max_depth = mother.hparams[0]
            min_samples_split = father.hparams[1]

        return DTR_Individual(hparams=[max_depth, min_samples_split], hparam_names=self.hparam_names,
                              model= DecisionTreeRegressor(criterion='squared_error', max_depth=max_depth,
                                                            min_samples_split=min_samples_split))
    
    def initialize_population(self, pop_size=400):
        for i in range(pop_size):
            self.population.add_individual(self.create_individual())

    def run_generation(self, x, y, display=True, window=None,
                       font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3,
                       font_color=(255, 255, 255), font_thickness=1, debug=False):

        x_values = [model.hparams[0] for model in self.population.individuals]
        y_values = [model.hparams[1] for model in self.population.individuals]
        x_diversity = stt.variance(x_values)
        y_diversity = stt.variance(y_values)
        GDI = stt.harmonic_mean([x_diversity, y_diversity])


        # Separates data into train and test groups
        x_train, x_test, y_train, y_test = tts(x, y, test_size=0.1)

        # Scales the data
        '''scaler = MinMaxScaler().fit(x_train)
        x_train = scaler.transform(x_train)

        scaler = MinMaxScaler().fit(x_test)
        x_test = scaler.transform(x_test)'''

        score_list = []
        n_list = []
        ind_list = []
        for n, individual in enumerate(self.population.individuals):
            if debug:
                print(individual)
            individual.model.fit(x_train, y_train)
            preds = individual.model.predict(x_test)
            individual.score = eval(f'{self.metric}(y_test, preds)') # Might raise an exeption
            score_list.append(individual.score)
            n_list.append(n)
            ind_list.append(individual)
            avg_score =  stt.mean(score_list)
            if display:
                window[:, :] = (0, 0, 0)
                cv2.putText(window, f'Generation: {self.population.generation} -- Average {self.metric}: {round(avg_score, 4)} -- n: {n} -- GDI: {GDI}',
                            (10, 20), font_face, font_scale, font_color, font_thickness)
                for z, ind in enumerate(ind_list):
                    cv2.circle(window, (112 + 2*z, 400 - min([int(ind.score), 256])), radius=2,
                            color=(ind.hparams[0] * 8 - 1, 255, ind.hparams[1] * 8 - 1),
                            thickness=-1)
                
                cv2.imshow('DTR_GA is running...', window)
                cv2.waitKey(1)

        self.population.rank(self.goal)
        self.metric_history.append(avg_score)

    def cross_breed(self, mutation=True, mutation_chance=0.02, mutation_strength=5):
        best_individuals = self.population.individuals[:40]
        mother_models = [best_individuals[k] for k in range(0, 40, 2)]
        father_models = [best_individuals[k] for k in range(1, 40, 2)]
        child_models = []

        for mother in mother_models:
            for father in father_models:
                child_model = self.create_individual([mother, father])
                child_models.append(child_model)
        for child_model in child_models:
            if random.random() <= mutation_chance:
                param_index = random.randint(0, 1)
                print(f'Mutation triggered at {self.hparam_names[param_index]} - Gen {self.population.generation}')
                child_model.hparams[param_index] = child_model.hparams[param_index] + (random.choice([-1, 1]) * mutation_strength)
                if child_model.hparams[param_index] > 32:
                    child_model.hparams[param_index] = 32
                elif child_model.hparams[param_index] < 2:
                    child_model.hparams[param_index] = 2
        
        self.population.update(child_models)

    def display_population(self, debug=True):
        x_values = [model.hparams[0] for model in self.population.individuals]
        y_values = [model.hparams[1] for model in self.population.individuals]
        fig = plt.figure(figsize=(16,9))
        plt.scatter(x_values, y_values)
        plt.title(f'DTR-GA - Generation {self.population.generation} - Hyperparameters')
        plt.grid()
        plt.xlabel(self.hparam_names[0])
        plt.ylabel(self.hparam_names[1])
        plt.show()
        if debug:
            for j in range(len(x_values)):
                print(x_values[j], y_values[j])

    def run_for(self, x, y, n_generations=50, mutation_chance=0.02, mutation_strength=5, debug=False):
        window = np.zeros((512, 1024, 3), dtype=np.uint8)
        for gen in range(n_generations):
            self.run_generation(x, y, display=True, window=window,debug=debug)
            self.cross_breed()
        self.display_population()
                





