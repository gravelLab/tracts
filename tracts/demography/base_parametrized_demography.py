from __future__ import annotations
import logging
import math
import numbers
from abc import ABC, abstractmethod
import numpy as np
import scipy
import scipy.optimize
from tracts.demography.parameter import ParamType, Parameter, DependentParameter

logger = logging.getLogger(__name__)


class BaseFounderEvent(ABC):

    def __init__(self, found_time, source_populations, remainder_population, end_time = None):
        if not source_populations:
            raise ValueError('Source populations cannot be empty.')
        if not remainder_population and end_time is None:
            raise ValueError('Remainder population cannot be empty in pulse founder event')
        
        if end_time is not None and remainder_population is not None:
            raise ValueError("In continuous founder event, there should be no remainder population")
        if remainder_population in source_populations:
            raise ValueError('Source population cannot be the same as remainder population.')
        
        self.found_time = found_time
        self.end_time = end_time
        self.source_populations = source_populations
        self.remainder_population = remainder_population



    @abstractmethod
    def execute(self, parametrized_demography: BaseParametrizedDemography, params):
        pass


class FounderEvent(BaseFounderEvent):

    def __init__(self, found_time, source_populations, remainder_population, end_time = None):
        super().__init__(found_time=found_time, source_populations=source_populations,
                         remainder_population=remainder_population, end_time=end_time)

    def execute(self, parametrized_demography: BaseParametrizedDemography, params):
        true_start_time = parametrized_demography.get_param_value(self.found_time, params)
        start_time = math.ceil(true_start_time) # a discretized value to create a matrix that can include the event.  
        frac_part_start = start_time - true_start_time
        
        migration_matrix = np.zeros((start_time + 1, len(parametrized_demography.population_indices)))

        

        if self.end_time is None:
            remaining_rate = 1.
            # Fraction of migrants that get repeated in the next generation,
            # to ensure continuous behaviour for fractional start times.
            
            # TODO: We have different rates for both sexes
            for source_population, rate_param in self.source_populations.items():
                rate = parametrized_demography.get_param_value(rate_param, params)
                migration_matrix[start_time, parametrized_demography.population_indices[source_population]] = rate
                migration_matrix[start_time - 1, parametrized_demography.population_indices[source_population]] = (
                                rate * frac_part_start)
                remaining_rate -= rate

            if remaining_rate < 0:
                logger.warning('Founding migration rates add up to more than 1')

            migration_matrix[
                start_time, parametrized_demography.population_indices[self.remainder_population]] = remaining_rate
            migration_matrix[start_time - 1, parametrized_demography.population_indices[
                self.remainder_population]] = remaining_rate * frac_part_start

        else: # Here we follow the logic of ContinuousEvent, modified for the first generation
            
            float_end_time = parametrized_demography.get_param_value(self.end_time, params)
         
            integer_end_time = math.ceil(float_end_time)
            total_rate = 0
            #breakpoint()
            for source_population, rate_param in self.source_populations.items():
                total_rate += parametrized_demography.get_param_value(rate_param, params)
            
            for source_population, rate_param in self.source_populations.items():
                rate = parametrized_demography.get_param_value(rate_param, params)
                
                migration_matrix[integer_end_time - 1, parametrized_demography.population_indices[source_population]] += (
                    rate * (integer_end_time - float_end_time))
        
                for t in range(integer_end_time, start_time-1):
                    migration_matrix[t, parametrized_demography.population_indices[source_population]] += rate

                #the first generation must sum to 1
                migration_matrix[start_time, parametrized_demography.population_indices[source_population]] += rate / total_rate 
                # The second generation does not need to sum to one. However, we want a continuously varying matrix. If true start time is 7.00001, or 6.999
                # we want the 7th generation to be an almost full replacement
            
                migration_matrix[start_time-1, parametrized_demography.population_indices[source_population]] += frac_part_start*(rate / total_rate)  + rate*(1- frac_part_start)


        return migration_matrix


class BaseMigrationEvent(ABC):

    def __init__(self, rate_parameter, source_population):
        self.rate_parameter = rate_parameter
        self.source_population = source_population

    @abstractmethod
    def execute(self, parametrized_demography: BaseParametrizedDemography, migration_matrix: np.ndarray, params):
        pass

class BaseParametrizedDemography(ABC):
    logger = logger

    def __init__(self, name: str = "", min_time=2, max_time=np.inf):
        self.name = name
        self.min_time = min_time
        self.max_time = max_time
        self.constraints = []
        self.model_base_params: dict[str, Parameter] = {}
        self.fixed_params: dict[str, Parameter] = {}
        self.dependent_params = {}
        self.constant_params = {}
        self.population_indices = {}
        self.reduced_constraints = []
        self.finalized = False
        self.founder_events: dict[str, FounderEvent]={}
        self.events: dict[str: list[BaseMigrationEvent]]={}        
        self.fixed_parameter_handler = FixedParametersHandler(self.logger)
        self.parametrized_populations= []



    @property
    def params_fixed_by_ancestry(self):
        return self.fixed_parameter_handler.params_fixed_by_ancestry
    
    @property
    def params_fixed_by_value(self):
        return self.fixed_parameter_handler.params_fixed_by_value
    @property
    def has_been_fixed(self):
        return self.fixed_parameter_handler.has_been_fixed

    @property
    def parameter_bounds(self):
        return [param.bound for param in self.model_base_params.values()]

    @staticmethod
    def proportions_from_matrix(migration_matrix: np.ndarray):
        current_ancestry_proportions = migration_matrix[-1, :]
        for row in migration_matrix[-2::-1, :]:
            current_ancestry_proportions = current_ancestry_proportions * (1 - row.sum()) + row
            if not np.isclose(current_ancestry_proportions.sum(), 1):
                raise ValueError('Current ancestry proportions do not sum to 1.')
        return current_ancestry_proportions
    
    def proportions_from_matrices(self, migration_matrices: dict[str, np.ndarray]):
        return {sample_pop: self.proportions_from_matrix(matrix) for sample_pop, matrix in migration_matrices.items()} 

    def proportions_from_matrices_return_keys(self):
        '''
        This method returns the expected keys from ``self.proportions_from_matrices()``.
        It is used by ``FixedParametersHandler`` to validate that the fixed parameter will be solvable from the given data.
        '''
        #TODO: calculate automatically from proportions_from_matrices().
        #For now, subclasses that change the behaviour of proportions_from_matrices() should have a different implementation of this method to reflect this.
        
        return set(self.founder_events.keys())

    def finalize(self):
        self.finalized = True
        for index, param_name in enumerate(self.model_base_params):
            self.model_base_params[param_name].index = index
        for index, population_name in enumerate(self.population_indices):
            self.population_indices[population_name] = index

    def add_parameter(self, param_name: str, param_type: ParamType=ParamType.UNTYPED, bounds=None):
        """
        Adds the given parameter name to the free parameters of the model. Free parameters include all parameters that can be optimized directly. 
        They may become fixed during optimization
        """
        self.finalized = False
        if param_name in self.model_base_params or param_name in self.dependent_params:
            self.logger.warning(f'Parameter "{param_name}" already exists.')
            return
        if bounds is None:
            if param_type == ParamType.TIME:
                bounds = (self.min_time, self.max_time)
            else:
                bounds = param_type.bounds
        self.model_base_params[param_name] = Parameter(param_name, param_type, bounds)


    def add_dependent_parameter(self, param_name: str, expression: function[["BaseParametrizedDemography",list[float]], float], param_type: ParamType=ParamType.UNTYPED, bounds=None):
        """Dependent parameters cannot be optimized directly. This is used in sex-biased migration to define a sex-specific migration rate computed from an overall migration
         rate and sex """
        if param_name in self.dependent_params:
            raise ValueError(f'Dependent parameter "{param_name}" already exists.')
        if bounds is None:
            if param_type == ParamType.TIME:
                bounds = (self.min_time, self.max_time)
            else:
                bounds = param_type.bounds
        self.dependent_params[param_name]=DependentParameter(param_name, expression, param_type, bounds)
        if param_name in self.model_base_params:
            self.model_base_params.pop(param_name)
        


    def add_population(self, population_name: str):
        """
        Adds the given population name to the populations of the model.
        """
        if self.fixed_parameter_handler.has_been_fixed:
            raise ValueError('Cannot add populations to a model after fixing ancestry proportions.')
        self.finalized = False
        if population_name not in self.population_indices:
            # population_indices will be given values when the model is finalized
            self.population_indices[population_name] = None

    def execute_migration_events(self, migration_matrix, params):
        for event in self.events:
            event.execute(self, migration_matrix, params)

    def get_index(self, time_param_name: str, population_name: str, params: list[float]):
        """
        Returns the matrix index as a tuple from the position and time. Reduces repetitive code.
        """

        return self.get_param_value(time_param_name, params), self.population_indices[population_name]

    def is_time_param(self):
        if not self.fixed_parameter_handler.has_been_fixed:
            return [param.type == ParamType.TIME for param in self.model_base_params.values()]
        time_param_list = []
        for param_name, param in self.model_base_params.items():
            if param_name not in self.fixed_parameter_handler.params_fixed_by_ancestry:
                time_param_list.append(param.type == ParamType.TIME)
        return time_param_list

    def get_param_value(self, param_name: str, params: list[float]):
        """
        Gets the correct value from the name of the parameter and the list of passed params.
        If *param_name* is a number instead, uses the number directly.
        """
        if isinstance(param_name, numbers.Number):
            return param_name
        if not self.finalized:
            raise ValueError('Cannot get parameter value before the model is finalized.')
        if param_name in self.model_base_params:
            return params[self.model_base_params[param_name].index]
        if param_name in self.constant_params:
            return self.constant_params[param_name].value
        if param_name in self.dependent_params:
            return self.dependent_params[param_name](self, params) 
        raise KeyError(f'Parameter "{param_name}" could not be found')

    def get_violation_score(self, params: list[float]):
        """
        Takes in a list of params equal to the length of ``model_base_params`` and returns a negative violation score if the resulting matrix would be or is invalid.
        """
        if self.fixed_parameter_handler.has_been_fixed:
            if len(params) != len(self.model_base_params):
                full_params = self.insert_params(params.copy(), [0 for _ in self.params_fixed_by_ancestry])
            else:
                full_params = params
            violation_score = min(self.check_bounds(full_params), self.check_constraints(full_params))
            if violation_score < 0:
                return violation_score
            params = self.fixed_parameter_handler.compute_params_fixed_by_ancestry(self, params)
        self.logger.info(f'Running bounds check.')
        violation_score = min(self.check_bounds(params), self.check_constraints(params))
        if violation_score < 0:
            return violation_score
        
        for migration_matrix in self.get_migration_matrices(params).values():
            #if np.min(migration_matrix)<0 :
            #   breakpoint()
            
            totmig = migration_matrix.sum(1).max()
            if 1 - totmig < violation_score:
                violation_score = 1 - totmig
                
            if np.min(migration_matrix)< violation_score:
                violation_score = min(migration_matrix)
        return violation_score

    def check_constraints(self, params: list[float]):
        """
        Constraints take the form of a dict ``{'param_subset':Tuple[String], 'expression': lambda (param_subset)}``.
        The violation score is the largest negative value from all the constraints.
        """
        violation_score = 0
        if not self.has_been_fixed:
            for constraint in self.constraints:
                violation = constraint['expression'](
                    [self.get_param_value(param_name, params) for param_name in constraint['param_subset']])
                if violation < violation_score:
                    violation_score = violation
                    self.logger.warning(f'{constraint["message"]} Out of bounds by: {-violation}.')
        else:
            if len(params) != len(self.model_base_params):
                full_params = self.insert_params(params.copy(), [0 for _ in self.params_fixed_by_ancestry])
            else:
                full_params = params
            for constraint in self.constraints:
                violation = constraint['expression'](
                    [self.get_param_value(param_name, full_params) for param_name in constraint['param_subset']])
                if violation < violation_score:
                    self.logger.warning(f'{constraint["message"]} Out of bounds by: {-violation}.')
                    violation_score = violation
        return violation_score

    def insert_params(self, params, params_from_proportions):
        '''
        Used for merging the parameters solved by the primary optimizer
        with the parameters found from the known ancestry proportions
        into a single list of parameters in the correct order for the model.
        '''
        if not self.params_fixed_by_ancestry:
            raise Exception("The insert_params method must be called only on fixed-proportion demographies")
        # self.logger.info(f'Params: {params}, params')
        if len(params_from_proportions) != len(self.params_fixed_by_ancestry):
            raise ValueError('Incorrect number of parameters to be solved')
        if len(params) + len(params_from_proportions) == len(self.model_base_params):
            iter_params = iter(params)
            iter_params_to_solve = iter(params_from_proportions)
            params = [next(iter_params_to_solve) if (param_name in self.params_fixed_by_ancestry) else next(iter_params)
                      for param_name in self.model_base_params]
            return params
        if len(params) == len(self.model_base_params):
            for param_name, value in zip(self.params_fixed_by_ancestry, params_from_proportions):
                params[self.model_base_params[param_name]['index']] = value
            return params
        raise ValueError('An unexpected error occured while merging parameters.')

    def check_bounds(self, params: list[float]):
        """
        Checks the bounds on parameters.
        Bounds should be absolute restrictions on possible parameter values,
        whereas constraints should be restrictions on parameter values relative to each other.
        """
        violation_score = 0
        if not self.fixed_parameter_handler.has_been_fixed:
            for param_name, param_object in self.model_base_params.items():
                violation = self.get_param_value(param_name, params) - param_object.bounds[0]
                if violation < violation_score:
                    self.logger.warning(
                        f'Lower bound for parameter {param_name} is {param_object.bounds[0]}. '
                        f'Out of bounds by: {-violation}.')
                    violation_score = violation
                violation = param_object.bounds[1] - self.get_param_value(param_name, params)
                if violation < violation_score:
                    self.logger.warning(
                        f'Upper bound for parameter {param_name} is {param_object.bounds[1]}. '
                        f'Out of bounds by: {-violation}.')
                    violation_score = violation
        else:
            if len(params) != len(self.model_base_params):
                full_params = self.insert_params(params.copy(), [0 for _ in self.params_fixed_by_ancestry])
            else:
                full_params = params
            # print(full_params, self.model_base_params)
 
            for param_name, param_object in self.model_base_params.items():
                #if param_name in self.params_fixed_by_ancestry:
                #    continue
                violation = self.get_param_value(param_name, full_params) - param_object.bounds[0]
                if violation < violation_score:
                    self.logger.warning(
                        f'Lower bound for parameter {param_name} is {param_object.bounds[0]}. '
                        f'Current value is {self.get_param_value(param_name, full_params)}.')
                    violation_score = violation
                violation = param_object.bounds[1] - self.get_param_value(param_name, full_params)
                if violation < violation_score:
                    self.logger.warning(
                        f'Upper bound for parameter {param_name} is {param_object.bounds[1]}. '
                        f'Current value is {self.get_param_value(param_name, full_params)}.')
                    violation_score = violation
        return violation_score

    @staticmethod
    def parse_proportions(ancestor_names: list[str], proportions: list[str]) -> tuple[dict[str:str], str]:
        """
        Parses the ancestry proportions used in a founding event into a dict of parametrized source populations
        and a remainder population.
        """
        #May later be folded into the add_founder_event() method.
        #TODO: add support for constants in proportions
        
        remainder_population = None
        remainder_proportion_string = None
        source_populations = {}
        population_and_proportion = zip(ancestor_names, proportions)
        for population, proportion in population_and_proportion:
            if isinstance(proportion, str) and proportion.startswith('1-'):
                if remainder_population is not None:
                    raise ValueError(
                        'More than one population detected whose proportion parameter begins with "1-".\n'
                        'This syntax is reserved for the population whose proportion is fixed'
                        ' by the proportions of the other populations '
                        'such that the sum of all proportions is 1.\n'
                        'Only one proportion should be an expression beginning with "1-"')
                remainder_population = population
                remainder_proportion_string = proportion
            else:
                if '-' in proportion:
                    raise ValueError('Parameter names cannot contain "-" when used in founding events.')
                source_populations.update({population: proportion})

        # Check that a remainder population was found
        assert remainder_population, ('The given proportions are not guaranteed to sum to 1.\n'
                                      'When using parametrized founding proportions, a population must be specified '
                                      'whose proportion takes the form "1-[the other proportions]".\n'
                                      'For example, in a three-population founder event, '
                                      'if two of the proportions are "a" and "b", the other must be "1-a-b"')

        # Check if the "1-" expression correctly contains all the other parameters.
        if not all(p1 == p2 for p1, p2 in
                   zip(source_populations.values(), remainder_proportion_string.split('-')[1:])):
            raise ValueError(
                'The given proportions are not guaranteed to sum to 1.\n'
                'When using parametrized founding proportions, a population must be specified '
                'whose proportion takes the form "1-[the other proportions]".\n'
                'For example, in a three-population founder event, if two of the proportions are "a" and "b",'
                ' the other must be "1-a-b"')

        return source_populations, remainder_population
    
    def list_parameters(self):
        for param_name, param_info in self.model_base_params.items():
            print(f"{param_name}: {param_info.type}")
        return

    def set_up_fixed_ancestry_proportions(self, params_to_fix_by_ancestry: list[str], proportions: dict[str: list[float]],
                                          params_to_fix_by_value:dict[str:float]={}):
        self.fixed_parameter_handler.set_up_fixed_ancestry_proportions(self, params_to_fix_by_ancestry, proportions, params_to_fix_by_value)

    @abstractmethod
    def get_random_parameters():
        pass

    @abstractmethod
    def get_migration_matrices(self, params: list[float], solve_using_known_proportions: bool = None) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def add_pulse_migration(self, source_population, rate_param, time_param):
        pass

    @abstractmethod
    def add_continuous_migration(self, source_population, rate_param, start_param, end_param):
        pass


class FixedParametersHandler:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.params_not_fixed_by_ancestry = []
        self.params_fixed_by_ancestry = {}
        self.known_ancestry_proportions: dict[str, list[float]] = None
        self.reduced_constraints =[]
        self.params_fixed_by_value = {}
        

    @property
    def has_been_fixed(self):
        return self.known_ancestry_proportions is not None

    def set_up_fixed_ancestry_proportions(self, demography: BaseParametrizedDemography, params_to_fix_by_ancestry: list[str], proportions: dict[str: list[float]],
                                          params_to_fix_by_value:dict[str:float]={}
                                          ):
        """
        Tells the model to calculate certain rate parameters based on the known
        ancestry proportions of the sample populations. Proportions are given as a dict with keys corresponding to the sample populations.

        We can fix parameters by value or to match global ancestry proportion. 
        """

        
        self.params_fixed_by_value = {param_name: params_to_fix_by_value[param_name] for param_name in demography.model_base_params if
                                         param_name in params_to_fix_by_value}
        demography.fixed_parameter_values = {params_to_fix_by_value[param_name] for param_name in demography.model_base_params if
                                         param_name in params_to_fix_by_value} # Store value in the demography object 
                                                                               # --these shoul dnot change     
        
        if not (demography.proportions_from_matrices_return_keys() == proportions.keys()):
            raise KeyError(
                "The keys of the provided sample proportions do not match proportions_from_matrices():"
                f"\nExpected keys: {demography.proportions_from_matrices_return_keys()}"
                f"\nProvided keys: {proportions.keys()}"
            )
        for param_name in params_to_fix_by_ancestry:
            if param_name in demography.dependent_params:
                    raise KeyError(f'{param_name} is already specified by another equation.')
            if param_name not in demography.model_base_params:
                raise KeyError(f'{param_name} is not a parameter of this model.')
            if param_name in demography.fixed_params:
                raise KeyError(f'{param_name} is already a fixed parameter.')
            if demography.model_base_params[param_name].type not in {ParamType.RATE, ParamType.SEX_BIAS}:
                raise ValueError(f'{param_name} is not a rate or sex bias parameter.')
        if len(params_to_fix_by_ancestry) != sum(len(prop)-1 for prop in proportions.values()):
            raise ValueError(
                    f'Number of parameters to fix is incorrect.'
                    f'Each population of interest can have N-1 proportions fixed'
                    f'Where N is the number of ancestral sources for that population'
                )
        
        # Use a dict to maintain order. Also looping over demography.model_base_params rather than params_to_fix to maintain order.
        self.params_fixed_by_ancestry = {param_name: '' for param_name in demography.model_base_params if
                                         param_name in params_to_fix_by_ancestry}
        
        # Exclude the last set of proportions because they are redundant (proportions must sum to 1)
        self.known_ancestry_proportions = {key:prop[:-1] for key, prop in proportions.items()}
        
        # Keep the constraints that involve any of the fixed parameters. Not used yet.
        self.reduced_constraints = [constraint for constraint in demography.constraints if any(
            param_name in self.params_fixed_by_ancestry for param_name in constraint['param_subset'])]
    

    def full_params_objective_func(self, parameters, demography):
        """returns the difference between computed and model ancestry proportions, as an array."""
        migration_matrices = demography.get_migration_matrices(
            parameters,
            solve_using_known_proportions=False)
        found_props = demography.proportions_from_matrices(migration_matrices)
        diff = np.array([found_props[ancestor][:-1] - self.known_ancestry_proportions[ancestor] 
                for ancestor in self.known_ancestry_proportions.keys()]).flatten()
        return diff

    
    def compute_params_fixed_by_ancestry(self, demography: BaseParametrizedDemography, params: list[float], known_ancestry_proportions=None):
        if not self.has_been_fixed and len(params) != len(demography.model_base_params):
            raise Exception("The demography has not been fixed yet.")
        if known_ancestry_proportions==None:
            known_ancestry_proportions=self.known_ancestry_proportions

        self.logger.info(f'Params before fixed-ancestry solving: {params}')
        if len(params) == len(demography.model_base_params): # Check if you are already satsifying the proportions
            full_params = params
            migration_matrices = demography.get_migration_matrices(full_params, solve_using_known_proportions=False)
            calculated_proportions = demography.proportions_from_matrices(migration_matrices)
            if np.all([np.allclose(calculated_proportions[sample_pop][:-1], known_ancestry_proportions[sample_pop])
                        for sample_pop in known_ancestry_proportions.keys()]):
                return full_params
        else:
            full_params = params.copy()


        def param_objective_func(params_to_solve):
            """Computes the difference as a function of the paramters to solve only. """
            nonlocal full_params
            params_to_solve[np.isnan(params_to_solve)] = 0
            full_params = self.insert_solved_params(demography.model_base_params, full_params, params_to_solve)
            
            return self.full_params_objective_func(full_params, demography)
        
        solved_params = scipy.optimize.fsolve(lambda params_to_solve: param_objective_func(params_to_solve),
                                              np.ones(len(self.params_fixed_by_ancestry)) * .2)

        full_params = self.insert_solved_params(demography.model_base_params, full_params, solved_params)
        self.logger.info(f'Params after solving with ancestry proportions: {full_params}')
        #if min(full_params) < -1:
        #    breakpoint()
        return full_params

    def insert_solved_params(self, model_base_params: dict[str, Parameter], params: list[float], params_from_proportions: list[float]):
        '''
        Used for merging the parameters solved by the primary optimizer
        with the parameters found from the known ancestry proportions
        into a single list of parameters in the correct order for the model.
        '''
        if not self.has_been_fixed:
            raise Exception("The insert_params method must be called only on demographies with known ancestry proportions")
        
        if len(params_from_proportions) != len(self.params_fixed_by_ancestry):
            raise ValueError('Incorrect number of parameters to be solved')
        
        # This is the case when 'params' contains only the completely free parameters, in order:
        if len(params) + len(params_from_proportions) == len(model_base_params):
            iter_params = iter(params)
            iter_params_to_solve = iter(params_from_proportions)
            params = [next(iter_params_to_solve) if (param_name in self.params_fixed_by_ancestry) else next(iter_params)
                      for param_name in model_base_params]
            return params
        
        # This is the case when 'params' contains both sets of parameters, in order,
        # And values corresponding to fixed parameters are to be replaced in the list.
        if len(params) == len(model_base_params):
            for param_name, value in zip(self.params_fixed_by_ancestry, params_from_proportions):
                params[model_base_params[param_name].index] = value
            return params
        raise ValueError('An unexpected error occured while merging parameters.'
                    f'\nNumber of model parameters: {len(model_base_params)}'
                    f'\nNumber of parameters provided: {len(params)}'
                    f'\nNumber of fixed parameters: {len(self.params_fixed_by_ancestry)}'
                )

    def insert_fixed_params(self, model_base_params: dict[str, Parameter], params_to_optimize: list[float], fixed_params: list[float]):
        '''
        Used for merging the parameters fixed parameters. This could be refactored with insert solved parameters
        with the parameters found from the known ancestry proportions
        
        non_computed_parameters include both parameters fixed by value and parameters to be optimized over. 
        params_to_optimize: the list of values for the parameters fixed by neither values nor ancestry proportions
        fixed_params: the values for the fixed parameters. 

        the list of values are presumed to be in the same order as in non-computed parameters

        Output: a list of values for all the non-computed parameters. This will then be used to compute the computed paramters. 

        '''
        
        assert (len(params_to_optimize) + len(fixed_params)+len(self.params_fixed_by_ancestry) == len(model_base_params)), f"{len(params_to_optimize)} + {len(fixed_params)}+{len(self.params_fixed_by_ancestry)} == {len(model_base_params)} non-computed parameters: {model_base_params}"
        

        iter_params = iter(params_to_optimize)
        iter_fixed_params = iter(fixed_params)
        params_to_optimize = [next(iter_fixed_params) if (param_name in self.params_fixed_by_value) else next(iter_params)  
                      for param_name in model_base_params if param_name not in self.params_fixed_by_ancestry ]
        return params_to_optimize
        



    def check_for_unsolvable_proportions(self, demography: BaseParametrizedDemography):
        '''
        Checks that the demography has an assignment of (full) parameters that results in the chosen proportions.
        '''
        def objective_func(params):
            migration_matrices = demography.get_migration_matrices(
                params,
                solve_using_known_proportions=False)
            diff = [prop[:-1] - self.known_ancestry_proportions[sample_pop] for sample_pop, prop in demography.proportions_from_matrices()]
            return np.linalg.norm(diff)
        
        result = scipy.optimize.minimize(objective_func, demography.get_random_parameters(), bounds=demography.parameter_bounds, constraints= {'ineq', demography.check_constraints})
        if not np.isclose(result.fun,0):
            raise ValueError(
                'The ancestry proportions in the sample are not achievable with the provided demographic model.')

    def check_for_improper_constraint(self, demography: BaseParametrizedDemography):
        '''
        Checks that the choice of parameters to fix does not underconstrain or overconstrain any of the matrices.
        '''
        starting_params = demography.get_random_parameters()
        target_matrices = demography.get_migration_matrices(starting_params)
        target_proportions = demography.proportions_from_matrices(target_matrices)
