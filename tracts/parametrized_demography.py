from __future__ import annotations
import numpy
import math
import ruamel.yaml
import itertools
import logging
import numbers
import scipy

'''
TODO: 
Add continuous migrations
Add fixed parameters and parameter equations
Add penalization function for non-resolvable constraints
Add function to solve constraints shape
Auto-create constraint function tyo give known ancestry fractions
Complete function to run the optimizer
'''
#global_logger = logging.getLogger(__name__)
class ParametrizedDemography:
    logger = logging.getLogger(__name__)

    '''
    A class representing a demographic history for a population, with parametrized migrations from other populations.
    TODO: add support for int (constant) parameters
    '''

    def __init__(self, name, min_time=2, max_time=numpy.inf):
        self.founder_event = None
        self.events = []
        self.constraints = []
        self.founding_time_param = ''
        self.name = name
        self.min_time = 2
        self.max_time = max_time

        self.params_fixed_by_ancestry = []
        self.params_not_fixed_by_ancestry = []
        self.free_params = {}
        self.dependent_params = {}
        self.constant_params = {}
        self.population_indices = {}
        self.reduced_constraints = []
        self.finalized = False
        return

    def add_parameter(self, param_name: str, type=None, bounds=None):
        '''
        Adds the given parameter name to the parameters of the model
        '''
        self.finalized = False
        if param_name not in self.dependent_params:
            if bounds == None:
                if type == 'time':
                    bounds = (self.min_time, self.max_time)
                if type == 'rate':
                    bounds = (0,1)
            self.free_params[param_name] = {'type':type, 'bounds':bounds}                

    def add_population(self, population_name: str):
        '''
        Adds the given population name to the populations of the model
        '''
        self.finalized = False
        self.population_indices[population_name] = None
        #population_indices will be given values when the model is finalized

    def get_param_value(self, param_name: str, params: list[float]):
        '''
        Gets the correct value from the name of the parameter and the list of passed params.
        If param_name is a number instead, uses the number directly
        '''
        if isinstance(param_name, numbers.Number):
            return param_name
        elif param_name in self.free_params:
            return params[self.free_params[param_name]['index']]
        elif param_name in self.constant_params:
            return self.constant_params[param_name]['value']
        elif param_name in self.dependent_params:
            return self.dependent_params[param_name](self, params)
        else:
            raise KeyError(f'Parameter "{param_name}" could not be found')
        
    def get_index(self, time_param_name: str, population_name: str, params: list[float]):
        '''
        Returns the matrix index as a tuple from the position and time. Reduces repetitive code
        '''
        
        return (self.get_param_value(time_param_name, params), self.population_indices[population_name])
    
    def is_time_param(self):
        return [param['type'] == 'time' for param in self.free_params.values()]

    def get_migration_matrix(self, params: list[float]) -> numpy.ndarray:
        '''
        Takes in a list of params equal to the length of free_params
        and returns a p*g migration matrix where p is the number of incoming populations and g is the number of generations
        If one of the parameters (time or migration) is incorrect, returns an empty matrix
        '''

        if self.finalized is not True:
            self.finalize()
        
        if len(params) != len(self.free_params):
            raise ValueError(f'Number of supplied parameters ({len(params)}) does not match the number of model parameters ({len(self.free_params)}).')

        if not self.founder_event:
            raise ValueError('Population is missing a founder event.') 

        migration_matrix = self.founder_event(self, params)

        for event in self.events:
            event(self, migration_matrix, params)
        return migration_matrix
    
    def fix_ancestry_proportions(self, params_to_fix, proportions) -> numpy.ndarray:
        '''
        Tells the model to calculate certain rate parameters based on the known ancestry proportions of the sample population
        '''

        for param_name in params_to_fix:
            if param_name not in self.free_params:
                if param_name in self.dependent_params:
                    raise KeyError(f'{param_name} is already specified by another equation.')
                raise KeyError(f'{param_name} is not a parameter of this model.')
            if self.free_params[param_name]['type'] != 'rate':
                raise ValueError(f'{param_name} is not a rate parameter.')
        if len(proportions) != len(self.population_indices):
            raise ValueError(f'Number of given ancestry proportions is not equal to the number of population indices.')
        if len(params_to_fix) != len(self.population_indices) - 1:
            raise ValueError(f'Number of parameters to fix is not equal to the number of population indices - 1.')
        self.__class__ = FixedAncestryDemography
        self.params_fixed_by_ancestry = {param_name:'' for param_name in self.free_params if param_name in params_to_fix}
        self.known_ancestry_proportions = proportions[:-1]
        self.reduced_constraints = [constraint for constraint in self.constraints if any(param_name in self.params_fixed_by_ancestry for param_name in constraint['param_subset'])]
        return
    
    def check_invalid(self, params: list[float]):
        '''
        Takes in a list of params equal to the length of free_params
        and returns a negative violation score if the resulting matrix would be or is invalid.
        '''
        self.logger.info(f'Running bounds check.')
        violation_score = min(self.check_bounds(params), self.check_constraints(params))
        if violation_score < 0:
            return violation_score
        migration_matrix = self.get_migration_matrix(params)
        for totmig in migration_matrix.sum(1):
            if 1 - totmig < violation_score:
                violation_score = 1 - totmig
        return violation_score
        
    def check_constraints(self, params: list[float]):
        '''
        Constraints take the form of a dict {'param_subset':Tuple[String], 'expression': lambda (param_subset)}
        The violation score is the largest negative value from all the constraints
        ''' 
        violation_score = 0
        for constraint in self.constraints:
            violation = constraint['expression']([self.get_param_value(param_name, params) for param_name in constraint['param_subset']])
            if violation < violation_score:
                violation_score = violation
                logging.warning(f'{constraint["message"]} Out of bounds by: {-violation}.')
        return violation_score

    def check_bounds(self, params: list[float]):
        '''
        Checks the bounds on parameters.
        Bounds should be absolute restrictions on possible parameter values,
        whereas Constraints should be restrictions on parameter values relative to each other.
        '''
        violation_score = 0
        for param_name, param_info in self.free_params.items():
            violation = self.get_param_value(param_name, params) - param_info['bounds'][0]
            if violation < violation_score:
                logging.warning(f'Lower bound for parameter {param_name} is {param_info["bounds"][0]}. Out of bounds by: {-violation}.')
                violation_score = violation
            violation = param_info['bounds'][1] - self.get_param_value(param_name, params)
            if violation < violation_score:
                logging.warning(f'Upper bound for parameter {param_name} is {param_info["bounds"][1]}. Out of bounds by: {-violation}.')
                violation_score = violation
        return violation_score


    def add_pulse_migration(self, source_population, rate_param, time_param):
        '''
        Adds a pulse migration from source population A, parametrized by time and rate
        '''

        self.add_population(source_population)
        self.add_parameter(rate_param, type='rate')
        self.add_parameter(time_param, type='time')

        self.constraints.append({
                'param_subset':(self.founding_time_param, time_param),
                'expression':lambda param_subset: param_subset[0] - param_subset[1] - 1,
                'message': 'Pulses cannot occur before or during the founding of the population.'
            })
        
        def _pulse_migration_event(self: 'ParametrizedDemography', migration_matrix: numpy.ndarray, params):
            t = self.get_param_value(time_param, params)
            a = self.get_param_value(rate_param, params)
            t2 = math.floor(t)
            #print(f'Pulse average time: {t}. Pulse start time: {t2}. Founding time: {self.get_param_value(self.founding_time_param, params)}')
            r2 = a*(t2+1-t)
            migration_matrix[t2, self.population_indices[source_population]] += r2
            migration_matrix[t2+1, self.population_indices[source_population]] += a*(t-t2)/(1-r2)
            return

        self.events.append(_pulse_migration_event)
        return
    
    def add_continuous_migration(self, source_population, rate_param, start_param, end_param):
        '''
        Adds a continuous migration from source population A, parametrized by start_time, end_time, and magnitude
        '''
        self.add_population(source_population)
        self.add_parameter(rate_param, type='rate')
        self.add_parameter(start_param, type='time')

        self.constraints.append({
                'param_subset':(self.founding_time_param, start_param),
                'expression':lambda param_subset: param_subset[0] - 1 - param_subset[1],
                'message': 'Migrations cannot start before or during the founding of the population.'
            })

        if end_param:
            self.add_parameter(end_param, type='time')
            self.constraints.append({
                'param_subset':(start_param, end_param),
                'expression':lambda param_subset: param_subset[0] - param_subset[1],
                'message': 'Migrations start time cannot be more recent than end time.'
            })

        def _continuous_migration_event(self: 'ParametrizedDemography', migration_matrix: numpy.ndarray, params):
            start_time = self.get_param_value(start_param, params)
            end_time = self.get_param_value(end_param, params) if end_param else 2
            t1 = math.floor(start_time)
            t2 = math.ceil(end_time)
            a = self.get_param_value(rate_param, params)
            
            migration_matrix[t1-1, self.population_indices[source_population]] += a*(end_time-t1)
            
            for t in range(t1, t2):
                migration_matrix[t, self.population_indices[source_population]] += a

            migration_matrix[t1, self.population_indices[source_population]] += a*(start_time-t1)
            migration_matrix[t2, self.population_indices[source_population]] += a*(t2-end_time)


        self.events.append(_continuous_migration_event)
        return

    def add_founder_event(self, source_populations: dict[str, str], remainder_population: str, found_time: str) -> None:
        '''
        Adds a founder event. A parametrized demography must have exactly one founder event.
        source_populations is a dict where each key is a population 
        and each value is the name of the parameter defining the migration ratio of each population
        remainder_population is the source of the remaining migrants, such that the total migration ratio adds up to 1.
        found_time is the name of the parameter defining the time of migration.
        '''

        if self.founder_event:
            raise ValueError('Population cannot have more than one founder event.')

        for population, rate_param in source_populations.items():
            self.add_population(population)
            self.add_parameter(rate_param, type='rate')

        self.add_population(remainder_population)

        self.add_parameter(found_time, type='time')
        self.founding_time_param = found_time
        
        def _founder_event(self: 'ParametrizedDemography', params):
            true_start_time = self.get_param_value(found_time, params)
            start_time = math.ceil(true_start_time)            
            migration_matrix = numpy.zeros((start_time+1, len(self.population_indices)))

            remaining_rate = 1
            
            #Fraction of migrants that get repeated in the next generation, to ensure continuous behaviour for fractional start times.
            repeated_migrant_fraction = start_time-true_start_time
            
            for population, rate_param in source_populations.items():
                rate = self.get_param_value(rate_param, params)
                migration_matrix[start_time, self.population_indices[population]] = rate
                migration_matrix[start_time-1, self.population_indices[population]] = rate*repeated_migrant_fraction
                remaining_rate -= rate
            
            if remaining_rate < 0:
                logging.warning('Founding migration rates add up to more than 1')
                
            migration_matrix[start_time, self.population_indices[remainder_population]] = remaining_rate
            migration_matrix[start_time - 1, self.population_indices[remainder_population]] = remaining_rate*repeated_migrant_fraction

            return migration_matrix

        self.founder_event = _founder_event
        return            

    def finalize(self):
        self.finalized = True
        for index, param_name in enumerate(self.free_params):
            self.free_params[param_name]['index'] = index
        for index, population_name in enumerate(self.population_indices):
            self.population_indices[population_name] = index
        return


    @staticmethod
    def proportions_from_matrix(migration_matrix):
        current_ancestry_proportions = migration_matrix[-1,:]
        for row in migration_matrix[-2::-1,:]:
            current_ancestry_proportions = current_ancestry_proportions*(1-row.sum())+row
            if not numpy.isclose(current_ancestry_proportions.sum(), 1):
                raise ValueError('Current ancestry proportions do not sum to 1.')
        return current_ancestry_proportions
    
    @staticmethod
    def load_from_YAML(filename: str) -> 'ParametrizedDemography':
        '''
        Creates an instance of ParametrizedDemography from a YAML file
        '''
        demography = ParametrizedDemography('')
        with open(filename) as file, ruamel.yaml.YAML(typ="safe") as yaml:
            demes_data = yaml.load(file)
            assert isinstance(demes_data, dict), ".yaml file was invalid."
            demography.name = demes_data['model_name'] if 'model_name' in demes_data else 'Unnamed Model'
            for population in demes_data['demes']:
                if 'ancestors' in population:
                    parametrized_population = population['name']
                    source_populations, remainder_population = ParametrizedDemography.parse_proportions(population['ancestors'], population['proportions'])
                    demography.add_founder_event(source_populations, remainder_population, population['start_time'])
            if 'pulses' in demes_data:
                for pulse in demes_data['pulses']:
                    if pulse['dest'] == parametrized_population:
                        for source, proportion in zip(pulse['sources'], pulse['proportions']):
                            demography.add_pulse_migration(source, proportion, pulse['time'])
            if 'migrations' in demes_data:
                for migration in demes_data['migrations']:
                    if 'dest' in migration and migration['dest'] == parametrized_population:
                        demography.add_continuous_migration(migration['source'], migration['rate'], migration['start_time'], migration['end_time'])
            demography.finalize()
        return demography
    
    @staticmethod
    def parse_proportions(ancestors: list[str], proportions: list)-> tuple[dict[str:str], list[str]]:
        '''
        Parses the ancestry proportions used in a founding event into a dict of parametrized source populations and a remainder population.
        May later be folded into the add_founder_event() method.
        TODO: add support for int arguments in proportions
        '''
        remainder_population = None
        remainder_proportion_string = None
        source_populations = {}
        for population, proportion in zip(ancestors, proportions):
            if isinstance(proportion, str) and proportion.startswith('1-'):
                assert remainder_population == None, ('More than one population detected whose proportion parameter begins with "1-".\n'
                    'This syntax is reserved for the population whose proportion is fixed by the proportions of the other populations '
                    'such that the sum of all proportions is 1.\n'
                    'Only one proportion should be an expression beginning with "1-"')
                remainder_population = population
                remainder_proportion_string = proportion
            else:
                assert '-' not in proportion, 'Parameter names cannot contain "-" when used in founding events.'
                source_populations.update({population:proportion})

        #Check that a remainder population was found
        assert remainder_population, ('The given proportions are not guaranteed to sum to 1.\n'
            'When using parametrized founding proportions, a population must be specified '
            'whose proportion takes the form "1-[the other proportions]".\n'
            'For example, in a three-population founder event, if two of the proportions are "a" and "b", the other must be "1-a-b"')
        
        #Check if the "1-" expression correctly contains all the other parameters.
        assert all(p1 == p2 for p1, p2 in zip(source_populations.values(), remainder_proportion_string.split('-')[1:])), ('The given proportions are not guaranteed to sum to 1.\n'
            'When using parametrized founding proportions, a population must be specified '
            'whose proportion takes the form "1-[the other proportions]".\n'
            'For example, in a three-population founder event, if two of the proportions are "a" and "b", the other must be "1-a-b"')

        return source_populations, remainder_population

class FixedAncestryDemography(ParametrizedDemography):
    '''
    Represents a parametrized demography with known final ancestry proportions
    '''

    def __init__(self):
        return
    
    
    def is_time_param(self):
        time_param_list = []
        for param_name, param in self.free_params.items():
            if param_name not in self.params_fixed_by_ancestry:
                time_param_list.append(param['type'] == 'time')
        return time_param_list
    
    def check_bounds(self, params: list[float]):
        '''
        Checks the bounds on parameters not fixed by ancestry.
        '''
        violation_score = 0
        if len(params) != len(self.free_params):
            full_params = self.insert_params(params.copy(), [0 for param in self.params_fixed_by_ancestry])
        else:
            full_params = params
        #print(full_params, self.free_params)
        for param_name, param_info in self.free_params.items():
            if param_name in self.params_fixed_by_ancestry:
                continue
            violation = self.get_param_value(param_name, full_params) - param_info['bounds'][0]
            if violation < violation_score:
                logging.warning(f'Lower bound for parameter {param_name} is {param_info["bounds"][0]}. Current value is {self.get_param_value(param_name, full_params)}.')
                violation_score = violation
            violation = param_info['bounds'][1] - self.get_param_value(param_name, full_params)
            if violation < violation_score:
                logging.warning(f'Upper bound for parameter {param_name} is {param_info["bounds"][1]}. Current value is {self.get_param_value(param_name, full_params)}.')
                violation_score = violation
        return violation_score

    def check_constraints(self, params: list[float]):
        '''
        Checks the bounds on parameters not fixed by ancestry.
        '''
        violation_score = 0
        if len(params) != len(self.free_params):
            full_params = self.insert_params(params.copy(), [0 for param in self.params_fixed_by_ancestry]) 
        else:
            full_params = params
        for constraint in self.reduced_constraints:
            violation = constraint['expression']([self.get_param_value(param_name, full_params) for param_name in constraint['param_subset']])
            if violation < violation_score:
                logging.warning(f'{constraint["message"]} Out of bounds by: {-violation}.')
                violation_score = violation
        return violation_score

    def check_invalid(self, params):
        '''
        Takes in a list of params equal to the length of free_params - params_fixed_by_ancestry
        and returns a negative violation score if the resulting matrix would be or is invalid.
        '''
        if len(params) != len(self.free_params):
            full_params = self.insert_params(params.copy(), [0 for param in self.params_fixed_by_ancestry]) 
        else:
            full_params = params
        violation_score = min(self.check_bounds(full_params), self.check_constraints(full_params))
        if violation_score < 0:
            return violation_score
        return super().check_invalid(self.get_full_params(params))

    def add_population(self, population_name: str):
        '''
        Adds the given population name to the populations of the model
        '''
        if population_name not in self.population_indices:
            raise ValueError('Cannot add populations to a model after fixing ancestry proportions.')
        self.population_indices[population_name] = None

    def get_full_params(self, params):
        self.logger.info(f'Params before fixed-ancestry solving: {params}')
        if len(params) == len(self.free_params):
            full_params = params
            if numpy.allclose(self.proportions_from_matrix(super().get_migration_matrix(full_params))[:-1], self.known_ancestry_proportions):
                return full_params
        else:
            full_params = params.copy()
        def _param_objective_func(self: FixedAncestryDemography, params_to_solve):
            nonlocal full_params
            params_to_solve[numpy.isnan(params_to_solve)] = 0
            full_params = self.insert_params(full_params, params_to_solve)
            #self.logger.info(f'Full params: {full_params}')
            found_props = self.proportions_from_matrix(super().get_migration_matrix(full_params))[:-1]
            fixed_props = self.known_ancestry_proportions
            diff = found_props-fixed_props
            return diff
        solved_params = scipy.optimize.fsolve(lambda params_to_solve: _param_objective_func(self, params_to_solve), numpy.ones(len(self.params_fixed_by_ancestry))*.2)
        full_params = self.insert_params(full_params, solved_params)
        self.logger.info(f'Params after solving with ancestry proportions: {full_params}')
        return full_params

    def get_migration_matrix(self, params):
        '''
        #print(super())
        #print(self.__class__.__mro__)
        self.logger.info(f'Inumpyut params: {params}')
        full_params = params.copy()
        self.logger.info('Full params: {full_params}')
        def _param_objective_func(self: FixedAncestryDemography, params_to_solve):
            params_to_solve[numpy.isnan(params_to_solve)] = 0
            self.logger.info('Full params: {full_params}')
            full_params = self.insert_params(full_params, params_to_solve)
            self.logger.info('Full params: {full_params}')
            found_props = self.proportions_from_matrix(super().get_migration_matrix(full_params))[:-1]
            fixed_props = self.known_ancestry_proportions
            diff = found_props-fixed_props
            return diff
        solved_params = scipy.optimize.fsolve(lambda params_to_solve: _param_objective_func(self, params_to_solve), (.2,))
        #print(solved_params)
        self.insert_params(full_params, solved_params)
        '''
        self.logger.info(f'Generating migration matrix.')
        return super().get_migration_matrix(self.get_full_params(params))
    
    def insert_params(self, params, params_to_solve):
        #self.logger.info(f'Params: {params}, params')
        if len(params_to_solve) != len(self.params_fixed_by_ancestry):
            raise ValueError('Incorrect number of parameters to be solved')
        if len(params) + len(params_to_solve) == len(self.free_params):
            iter_params = iter(params)
            iter_params_to_solve = iter(params_to_solve)
            params = [next(iter_params_to_solve) if (param_name in self.params_fixed_by_ancestry) else next(iter_params) for param_name in self.free_params]
            return params
        if len(params) == len(self.free_params):
            for param_name, value in zip(self.params_fixed_by_ancestry, params_to_solve):
                params[self.free_params[param_name]['index']] = value
            return params
        raise ValueError('Parameters fixed by ancestry proportions could not be resolved with the given parameters.')

def test():
    model = ParametrizedDemography()
    model.add_founder_event({'A': 'm1_A'}, 'B', 't0')
    model.add_pulse_migration('C', 'r1', 't1')
    model.add_pulse_migration('D', 'r1', 't2')
    model.finalize()
    m = model.get_migration_matrix([0.1, 4, 0.2, 1, 2.2])
    print(m)
    print(model.free_params)

def test_2():
    model = ParametrizedDemography.load_from_YAML('pp_px.yaml')
    m = model.get_migration_matrix([0.2, 4, 0.375, 3])
    print(m)
    print(model.proportions_from_matrix(m))
    print(model.free_params)
    model.fix_ancestry_proportions('r', [0.5, 0.5])
    print(model.params_fixed_by_ancestry)
    model.get_migration_matrix([0.2, 4, 0.375])

def test_3():
    model = ParametrizedDemography.load_from_YAML('pp.yaml')
    m = model.get_migration_matrix([0.2, 4.1])
    print(m)

if __name__ == '__main__':
    pass
