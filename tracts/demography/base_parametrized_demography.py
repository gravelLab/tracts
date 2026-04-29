from __future__ import annotations
import math
import numbers
from abc import ABC, abstractmethod
import numpy as np
import scipy
import scipy.optimize
from tracts.demography.parameter import ParamType, Parameter, DependentParameter
from typing import Callable
import logging
logger = logging.getLogger(__name__)

class BaseFounderEvent(ABC):
    """
    Base class for founder events. A founder event is an event in which a new population is formed from one or more source populations. The source populations contribute to the new population according to certain proportions, which can be fixed or parametrized.

    Attributes
    ----------
    found_time: str
        The name of the parameter defining the time of the founder event. 
    source_populations: dict[str, str]
        A dict mapping source population names to their contribution proportions. Proportions should sum to 1, but this is not enforced by the model. If they do not sum to 1, the remainder will be assigned to the remainder population.
    remainder_population: str
        The population that contributes the remaining proportion to the new population after the source populations have contributed according to their specified proportions. This population can be one of the source populations or a different population.
    end_time: str | None
        The name of the parameter defining the end time of the founder event. If None, the founder event is a pulse event. If not None, the founder event is a continuous event that starts at found_time and ends at end_time. In a continuous founder event, the migration rates are constant between ``found_time`` and ``end_time``, and the total migration rate is 1 at ``found_time`` and 0 at ``end_time``. In a pulse founder event, the migration rates are applied only at ``found_time``.
    """

    def __init__(self, found_time: str, source_populations: dict[str, str], remainder_population: str, end_time: str | None = None):
        """
        Initializes the founder event. The founder event is defined by the time of the event, the source populations and their contribution proportions, the remainder population, and the end time of the event (if it is a continuous event).

        Parameters
        ----------
        found_time: str
            The name of the parameter defining the time of the founder event.
        source_populations: dict[str, str]
            A dict mapping source population names to their contribution proportions. Proportions should sum to 1, but this is not enforced by the model. If they do not sum to 1, the remainder will be assigned to the remainder population.
        remainder_population: str
            The population that contributes the remaining proportion to the new population after the source populations have contributed according to their specified proportions. This population can be one of the source populations or a different population.
        end_time: str | None
            The name of the parameter defining the end time of the founder event. If None, the founder event is a pulse event. If not None, the founder event is a continuous event that starts at ``found_time`` and ends at ``end_time``. In a continuous founder event, the migration rates are constant between ``found_time`` and ``end_time``, and the total migration rate is 1 at ``found_time`` and 0 at ``end_time``. In a pulse founder event, the migration rates are applied only at ``found_time``.
        """

        if not source_populations:
            raise ValueError('Source populations cannot be empty.')
        if not remainder_population and end_time is None:
            raise ValueError('Remainder population cannot be empty in pulse founder event.')
        
        if end_time is not None and remainder_population is not None:
            raise ValueError("In continuous founder event, there should be no remainder population.")
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
    """
    A class that sets up a founder event. See :class:`~tracts.demography.base_parametrized_demography.BaseFounderEvent` for details.

    Attributes
    ----------
    found_time: str
        The name of the parameter defining the time of the founder event.
    source_populations: dict[str, str]
        A dict mapping source population names to their contribution proportions.
    remainder_population: str
        The population that contributes the remaining proportion to the new population after the source populations have contributed according to their specified proportions.
    end_time: str | None
        The name of the parameter defining the end time of the founder event. If None, the founder event is a pulse event. If not None, the founder event is a continuous event that starts at ``found_time`` and ends at ``end_time``. 
    """

    def __init__(self, found_time: str, source_populations: dict[str, str], remainder_population: str, end_time: str | None = None):
        """
        Initializes the founder event. 

        Parameters
        ----------
        found_time: str
            The name of the parameter defining the time of the founder event.
        source_populations: dict[str, str]
            A dict mapping source population names to their contribution proportions.
        remainder_population: str
            The population that contributes the remaining proportion to the new population after the source populations have contributed according to their specified proportions.
        end_time: str | None
            The name of the parameter defining the end time of the founder event. If None, the founder event is a pulse event. If not None, the founder event is a continuous event that starts at ``found_time`` and ends at ``end_time``.
        """

        super().__init__(found_time=found_time,
                        source_populations=source_populations,
                        remainder_population=remainder_population,
                        end_time=end_time)

    def execute(self, parametrized_demography: BaseParametrizedDemography, params):
        """
        Executes the founder event by modifying the migration matrix of the demography according to the parameters of the founder event. 

        Parameters
        ----------
        parametrized_demography: BaseParametrizedDemography
            The demography object that the founder event is being executed on. The migration matrix of the demography will be modified according to the parameters of the founder event.
        params: list[float]
            The list of parameters for the founder event.

        Returns
        -------
        np.ndarray
            The migration matrix of the demography after the founder event has been executed.
        """

        true_start_time = parametrized_demography.get_param_value(param_name=self.found_time,
                                                                params=params)
        start_time = math.ceil(true_start_time) # A discretized value to create a matrix that can include the event.  
        if true_start_time < 1:
            start_time_msg = f"Founder event time must be at least 1 generation ago. Current start time is {true_start_time}."
            logger.warning(start_time_msg + "If this happens at start of simulation, it may be a problem with parameter scaling.")
            raise ValueError(start_time_msg)

        frac_part_start = start_time - true_start_time
        migration_matrix = np.zeros((start_time + 1, len(parametrized_demography.population_indices)))        

        if self.end_time is None:
            remaining_rate = 1.  # Fraction of migrants that get repeated in the next generation, to ensure continuous behaviour for fractional start times.
            
            for source_population, rate_param in self.source_populations.items():
                rate = parametrized_demography.get_param_value(param_name=rate_param,
                                                                params=params)
                migration_matrix[start_time, parametrized_demography.population_indices[source_population]] = rate
                migration_matrix[start_time - 1, parametrized_demography.population_indices[source_population]] = rate * frac_part_start
                remaining_rate -= rate

            if remaining_rate < 0:
                logger.warning(f"Founding migration rates add up to more than 1 at params {params}, matrix {migration_matrix}.")

            migration_matrix[start_time, parametrized_demography.population_indices[self.remainder_population]] = remaining_rate
            migration_matrix[start_time - 1, parametrized_demography.population_indices[self.remainder_population]] = remaining_rate * frac_part_start

        else: # Follow the logic of ContinuousEvent, modified for the first generation.
            
            float_end_time = parametrized_demography.get_param_value(param_name=self.end_time,
                                                                    params=params)
            
            integer_end_time = math.ceil(float_end_time)
            total_rate = 0

            for source_population, rate_param in self.source_populations.items():
                total_rate += parametrized_demography.get_param_value(param_name=rate_param,
                                                                    params=params)
            
            for source_population, rate_param in self.source_populations.items():
                rate = parametrized_demography.get_param_value(param_name=rate_param,
                                                                params=params)
                migration_matrix[integer_end_time - 1, parametrized_demography.population_indices[source_population]] += rate * (integer_end_time - float_end_time)
        
                for t in range(integer_end_time, start_time-1):
                    migration_matrix[t, parametrized_demography.population_indices[source_population]] += rate

                migration_matrix[start_time, parametrized_demography.population_indices[source_population]] += rate / total_rate # First generation must sum to 1
                # The second generation does not need to sum to one. However, we want a continuously varying matrix. If true start time is 7.00001, or 6.999, we want the 7th generation to be an almost full replacement.
                migration_matrix[start_time-1, parametrized_demography.population_indices[source_population]] += frac_part_start*(rate / total_rate)  + rate*(1- frac_part_start)

        return migration_matrix


class BaseMigrationEvent(ABC):
    """
    Base class for migration events. A migration is an event in which a certain fraction of an admixed population is replaced with migrants from a given source population. Migration events can be pulse events, in which the migration happens at a single time point, or continuous events, in which the migration happens over a time interval with a constant migration rate.

    Attributes
    ----------
    rate_param: str
        The name of the parameter defining the migration rate of the migration event. In a pulse migration event, this is the fraction of the admixed population that is replaced with migrants from the source population. In a continuous migration event, this is the fraction of the admixed population that is replaced with migrants from the source population per generation.
    source_population: str
        The population that contributes migrants to the admixed population in the migration event.
    """
    
    def __init__(self, rate_parameter: str, source_population: str):
        """
        Initializes the migration event. The migration event is defined by the source population and the migration rate. 

        Parameters
        ----------
        rate_parameter: str
            The name of the parameter defining the migration rate of the migration event. 
        source_population: str
            The population that contributes migrants to the admixed population in the migration event.
        """
        self.rate_parameter = rate_parameter
        self.source_population = source_population

    @abstractmethod
    def execute(self, parametrized_demography: BaseParametrizedDemography, migration_matrix: np.ndarray, params):
        pass

class BaseParametrizedDemography(ABC):
    r"""
    Base class for parametrized demographies. A parametrized demography is a demographic model in which the migration events are defined by a set of parameters (that are optimized during inference).
    The class includes methods for adding migration events, checking the validity of the resulting migration matrices, and calculating the ancestry proportions resulting from the migration events. 
    The class also allows fixing certain parameters based on known ancestry proportions of the sample populations.

    Attributes
    ----------
    name: str 
        The name of the parametrized demography.
    min_time: int
        The minimum time for time parameters in the model. This is used to set the bounds for time parameters.
    max_time: int
        The maximum time for time parameters in the model. This is used to set the bounds for time parameters.
    constraints: list[dict]
        A list of constraints on the parameters of the model. Constraints are of the form :math:`g(\theta) \geq 0`, where :math:`\theta` denotes the model parameters. ``constraint`` is a dict with keys ``param_subset``, ``expression``, and ``message``. ``param_subset`` is a tuple of parameter names that are involved in the constraint (:math:`\theta`). ``expression`` is a lambda function representing :math:`g` and ``message`` is a string that describes the constraint, which is used for logging when the constraint is violated.
    model_base_params: dict[str, Parameter]
        A dict mapping parameter names to :class:`~tracts.parameter.Parameter` objects for the free parameters of the model. Free parameters are parameters that can be optimized directly. They may become fixed during optimization if they are fixed by ancestry proportions.
    fixed_params: dict[str, Parameter]
        A dict mapping parameter names to :class:`~tracts.parameter.Parameter` objects for the parameters that have been fixed by ancestry proportions. These parameters cannot be optimized directly and are computed from the ancestry proportions.
    dependent_params: dict[str, DependentParameter]
        A dict mapping parameter names to :class:`~tracts.parameter.DependentParameter` objects for the parameters that are dependent on other parameters.
    constant_params: dict[str, Parameter]
        A dict mapping parameter names to :class:`~tracts.parameter.Parameter` objects for the parameters that are fixed by the user to a certain value. These parameters cannot be optimized and are not computed from ancestry proportions.
    reduced_constraints: list[dict]
        A list of constraints that have been reduced from the original constraints after fixing parameters by ancestry proportions. These constraints are of the same form as the original constraints, but they only involve the free parameters that are being optimized after fixing parameters by ancestry proportions.
    population_indices: dict[str, int]
        A dict mapping population names to their corresponding indices in the migration matrices.
    finalized: bool
        A boolean indicating whether the model has been finalized. The model is finalized after all parameters and populations have been added and the indices for the parameters and populations have been set. The model must be finalized before it can be used for inference.
    founder_events: dict[str, FounderEvent]
        A dict mapping the names of founder events to their corresponding :class:`~tracts.demography.base_parametrized_demography.FounderEvent` objects.
    events: dict[str, list[BaseMigrationEvent]]
        A dict mapping population names to a list of migration events that affect that population. Each migration event is a :class:`~tracts.demography.base_parametrized_demography.BaseMigrationEvent` object.
    parameter_handler: FixedParametersHandler
        A :class:`~tracts.demography.base_parametrized_demography.FixedParametersHandler` object that handles the fixing of parameters based on ancestry proportions.
    parametrized_populations: list[str]
        A list of population names that are affected by migration events. This is used to determine which populations are included in the migration matrices and which populations' ancestry proportions are calculated from the migration matrices.
    logger: logging.Logger
        The logger.
    """

    def __init__(self, name: str | None = None, min_time: float = 1, max_time: float = np.inf):
        """
        Initializes the :class:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography` class. 

        Parameters
        ----------
        name: str | None
            The name of the parametrized demography. If None, the name is set to an empty string.
        min_time: float
            The minimum time for time parameters in the model. This is used to set the bounds for time parameters. 
        max_time: float
            The maximum time for time parameters in the model. This is used to set the bounds for time parameters. 
        """

        self.name = "" if name is None else name
        self.min_time: float = min_time
        self.max_time: float = max_time
        self.constraints: list[dict] = []
        self.model_base_params: dict[str, Parameter] = {}
        self.fixed_params: dict[str, Parameter] = {}
        self.dependent_params: dict[str, Parameter] = {}
        self.constant_params: dict[str, Parameter] = {}
        self.population_indices: dict[str, int] = {}
        self.reduced_constraints: list[dict] = []
        self.finalized: bool = False
        self.founder_events: dict[str, FounderEvent] = {}
        self.events: dict[str, list[BaseMigrationEvent]] = {}
        self.logger = logger
        self.parameter_handler: FixedParametersHandler = FixedParametersHandler(self.logger)
        self.parametrized_populations: list[str] = []

    @property
    def params_fixed_by_ancestry(self):
        """
        Gets the names of the parameters that are fixed by ancestry proportions.

        Returns
        -------
        dict[str, str]
            A dict of the names of the parameters that are fixed by ancestry proportions and their corresponding values.
        """
        return self.parameter_handler.params_fixed_by_ancestry
    
    @property
    def params_fixed_by_value(self):
        """
        Gets the names of the parameters that are fixed by user-defined values.

        Returns
        -------
        dict[str, str]
            A dict of the names of the parameters that are fixed by user-defined values and their corresponding values.
        """
        return self.parameter_handler.user_params_fixed_by_value
    
    @property
    def has_been_fixed(self):
        """
        Whether any parameters have been fixed by ancestry proportions.
        
        Returns
        -------
        bool
            True if any parameters have been fixed by ancestry proportions, False otherwise.
        """
        return self.parameter_handler.has_been_fixed

    @property
    def parameter_bounds(self):
        """
        Gets the bounds for the free parameters of the model.

        Returns
        -------
        list[tuple[float, float]]
            A list of tuples representing the bounds for the free parameters of the model. The order of the bounds corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        """
        return [param.bounds for param in self.model_base_params.values()]

    @staticmethod
    def proportions_from_matrix(migration_matrix: np.ndarray):
        r"""
        Takes in a migration matrix and returns the ancestry proportions resulting from the migration events represented by the matrix. 

        Parameters
        ----------
        migration_matrix: np.ndarray
            A migration matrix representing the migration events. The matrix should have dimensions :math:`(T, P)`, where :math:`T` is the number of time points and :math:`P` is the number of populations. Each entry in the matrix represents the fraction of the admixed population that is replaced by migrants from a source population at a given time point.
        
        Returns
        -------
        np.ndarray
            An array of shape :math:`(P,)` representing the ancestry proportions resulting from the migration events represented by the migration matrix.
        """
        current_ancestry_proportions = migration_matrix[-1, :]
        for row in migration_matrix[-2::-1, :]:
            current_ancestry_proportions = current_ancestry_proportions * (1 - row.sum()) + row
            if not np.isclose(current_ancestry_proportions.sum(), 1):
                raise ValueError('Current ancestry proportions do not sum to 1.')
        return current_ancestry_proportions
    
    def proportions_from_matrices(self, migration_matrices: dict[str, np.ndarray]):
        """
        Calculates the ancestry proportions for a set of migration matrices, each describing the migration events affecting a particular sample population.

        Parameters
        ----------
        migration_matrices: dict[str, np.ndarray]
            A dict mapping sample population names to their corresponding migration matrices.

        Returns
        -------
        dict[str, np.ndarray]
            A dict mapping sample population names to their corresponding ancestry proportions resulting from the migration events represented by the migration matrices.
        """        
        return {sample_pop: self.proportions_from_matrix(migration_matrix=matrix) for sample_pop, matrix in migration_matrices.items()} 

    def proportions_from_matrices_return_keys(self):
        """
        This method returns the expected keys from :func:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.proportions_from_matrices()`.
        It is used by :class:`~tracts.demography.base_parametrized_demography.FixedParametersHandler` to validate that the fixed parameter will be solvable from the given data.
        
        Returns
        -------
        set[str]
            The expected keys from :func:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.proportions_from_matrices()`, which are the names of the sample populations that are affected by migration events.
        """
        #TODO: calculate automatically from proportions_from_matrices(). For now, subclasses that change the behaviour of proportions_from_matrices() should have a different implementation of this method to reflect this.
        return set(self.founder_events.keys())

    def finalize(self):
        """
        Finalizes the model by setting the indices for the parameters and populations. This should be called after all parameters and populations have been added to the model and before the model is used for inference.
        """
        self.finalized = True
        for index, param_name in enumerate(self.model_base_params):
            self.model_base_params[param_name].index = index
        for index, population_name in enumerate(self.population_indices):
            self.population_indices[population_name] = index

    def add_parameter(self, param_name: str, param_type: ParamType=ParamType.UNTYPED, bounds: tuple | None = None):
        """
        Adds the given parameter name to the free parameters of the model. Free parameters include all parameters that can be optimized directly, and may become fixed during optimization.
        
        Parameters
        ----------
        param_name: str
            The name of the parameter to be added.
        param_type: ParamType
            The type of the parameter, which is used to set the default bounds for the parameter if bounds are not provided. If ParamType.UNTYPED is used, bounds must be provided.
        bounds: tuple | None
            A tuple representing the bounds for the parameter. If None, the bounds are set to the default bounds for the parameter type. If ParamType.UNTYPED is used, bounds must be provided.
        """
        self.finalized = False
        if param_name in self.model_base_params or param_name in self.dependent_params:
            self.logger.info(f'Parameter "{param_name}" already exists.')
            return
        if bounds is None:
            if param_type == ParamType.TIME:
                bounds = (self.min_time, self.max_time)
            else:
                bounds = param_type.bounds
        self.model_base_params[param_name] = Parameter(param_name, param_type, bounds)


    def add_dependent_parameter(self, param_name: str, expression: Callable[["BaseParametrizedDemography",list[float]], float], param_type: ParamType=ParamType.UNTYPED, bounds: tuple | None = None):
        """
        Dependent parameters cannot be optimized directly. This is used in sex-biased migration to define a sex-specific migration rate computed from an overall migration rate and sex.

        Parameters
        ----------
        param_name: str
            The name of the parameter to be added.
        expression: Callable[["BaseParametrizedDemography",list[float]], float]
            A function that computes the value of the dependent parameter based on other parameters.
        param_type: ParamType
            The type of the parameter, which is used to set the default bounds for the parameter if bounds are not provided. If ParamType.UNTYPED is used, bounds must be provided.
        bounds: tuple | None
            A tuple representing the bounds for the parameter. If None, the bounds are set to the default bounds for the parameter type. If ParamType.UNTYPED is used, bounds must be provided.
        """
        if param_name in self.dependent_params:
            raise ValueError(f'Dependent parameter "{param_name}" already exists.')
        if bounds is None:
            if param_type == ParamType.TIME:
                bounds = (self.min_time, self.max_time)
            else:
                bounds = param_type.bounds
        self.dependent_params[param_name]=DependentParameter(name=param_name,
                                                            expression=expression,
                                                            param_type=param_type,
                                                            bounds=bounds)
        if param_name in self.model_base_params:
            self.model_base_params.pop(key=param_name)
    

    def add_population(self, population_name: str):
        """
        Adds the given population name to the populations of the model.

        Parameters
        ----------
        population_name: str
            The name of the population to be added.
        """
        if self.parameter_handler.has_been_fixed:
            raise ValueError('Cannot add populations to a model after fixing ancestry proportions.')
        self.finalized = False
        if population_name not in self.population_indices:
            self.population_indices[population_name] = None # Population_indices will be given values when the model is finalized.

    def execute_migration_events(self, migration_matrix, params):
        for event in self.events:
            event.execute(self, migration_matrix, params)

    def get_index(self, time_param_name: str, population_name: str, params: list[float]):
        """
        Returns the matrix index as a tuple from the position and time. Reduces repetitive code.

        Returns
        -------
        tuple[float, int]
            A tuple of the form (time, population_index) representing the index in the migration matrix corresponding to the given time parameter and population name.
        """

        return self.get_param_value(param_name=time_param_name,
                                    params=params), self.population_indices[population_name]

    def is_time_param(self):
        """
        Determines if each parameter is a time parameter.
        
        Returns
        -------
        list[bool]
            A list of booleans indicating whether each parameter is a time parameter. The order of the booleans corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        """
        return [param.type == ParamType.TIME for param in self.model_base_params.values()]
       

    def get_param_value(self, param_name: str | float, params: list[float]):
        """
        Gets the correct value from the name of the parameter and the list of passed ``params``. If ``param_name`` is a number instead, uses the number directly.

        Parameters
        ----------
        param_name: str | float
            The name of the parameter to get the value for, or a float representing the value directly.
        params: list[float]
            The list of parameter values for the free parameters of the model. The order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
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
        raise KeyError(f'Parameter "{param_name}" could not be found.')
    

    def get_violation_score(self, params: list[float], verbose: bool = False):
        """
        Takes in a list of params equal to the length of :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params` and returns a negative violation score if the resulting matrix would be or is invalid.

        Parameters
        ----------
        params: list[float]
            The list of parameter values for the free parameters of the model. The order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        verbose: bool
            If True, logs the bound score, constraint score, and migration matrix violation scores when a violation is detected.
        
        Returns
        -------
        float
            A violation score, which is negative if the resulting migration matrix would be or is invalid and non-negative otherwise. The violation score is the largest negative value from the bound violations, constraint violations, and migration matrix violations.
        """
        if self.parameter_handler.has_been_fixed:
            if len(params) != len(self.model_base_params):
                full_params = self.insert_params(params.copy(), [0 for _ in self.params_fixed_by_ancestry])
            else:
                full_params = params
            violation_score = min(self.check_bounds(full_params), self.check_constraints(full_params))
            if violation_score < 0:
                return violation_score
            params = self.parameter_handler.compute_params_fixed_by_ancestry(params=params)
        self.logger.debug(f'Running bounds check.')
        bound_score =  self.check_bounds(params)   
        constraint_score = self.check_constraints(params)

        violation_score = min(bound_score, constraint_score)
        if violation_score < 0:
            if verbose:
                self.logger.debug(f'Violation detected: bound score={bound_score}, constraint score={constraint_score}')
            return violation_score
        
        for migration_matrix in self.get_migration_matrices(params).values():
            
            totmig = migration_matrix.sum(1).max()
            sum_over_1_violation = 1-totmig
            if sum_over_1_violation < violation_score:
                violation_score = sum_over_1_violation

            positive_violation = np.min(migration_matrix)
            if positive_violation < violation_score:
                violation_score = positive_violation
            if verbose and violation_score <0:
                    self.logger.debug(f'Violation detected: bound score : {bound_score}, constraint score : {constraint_score}, sum over 1 violation : {sum_over_1_violation}, positive_violation : {positive_violation}.')
                    self.logger.debug(f'Migration matrix:\n{migration_matrix}.')
        return violation_score

    def check_constraints(self, params: list[float]):
        """
        Checks the constraints on parameters. Constraints take the form of a dict ``{'param_subset':tuple[str], 'expression': lambda (param_subset), 'message': str}`` (see ``constraints`` in :class:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography`). 

        Parameters
        ----------
        params: list[float]
            The list of parameter values for the free parameters of the model. The order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        
        Returns
        -------
        float
            A violation score, which is negative if any of the constraints are violated and non-negative otherwise. The violation score is the largest negative value from all the constraints.
        """
        violation_score = 0
        if not self.has_been_fixed:
            for constraint in self.constraints:
                violation = constraint['expression'](
                    [self.get_param_value(param_name=param_name,
                                        params=params) for param_name in constraint['param_subset']])
                if violation < violation_score:
                    violation_score = violation
                    self.logger.debug(f'{constraint["message"]} Out of bounds by: {-violation}.')
        else:
            if len(params) != len(self.model_base_params):
                full_params = self.insert_params(params.copy(), [0 for _ in self.params_fixed_by_ancestry])
            else:
                full_params = params
            for constraint in self.constraints:
                violation = constraint['expression'](
                    [self.get_param_value(param_name=param_name,
                                        params=full_params) for param_name in constraint['param_subset']])
                if violation < violation_score:
                    self.logger.debug(f'{constraint["message"]} Out of bounds by: {-violation}.')
                    violation_score = violation
        return violation_score

    def insert_params(self, params: list[float], params_from_proportions: list[float]):
        """
        Merges the parameters solved by the primary optimizer with the parameters found from the known ancestry proportions into a single list of parameters in the correct order for the model.

        Parameters
        ----------
        params: list[float]
            The list of parameter values for the free parameters of the model that are being optimized by the primary optimizer. The order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params` that are not fixed by ancestry proportions.
        params_from_proportions: list[float]
            The list of parameter values for the parameters that are fixed by ancestry proportions, which are computed from the known ancestry proportions. The order of the values corresponds to the order of the parameters in ``params_fixed_by_ancestry``.
        
        Returns
        -------
        list[float]
            A list of parameter values for all the free parameters of the model, in the correct order.
        """
        if not self.params_fixed_by_ancestry:
            raise Exception("The insert_params method must be called only on fixed-proportion demographies.")
        
        if len(params_from_proportions) != len(self.params_fixed_by_ancestry):
            raise ValueError('Incorrect number of parameters to be solved.')
        
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
        Checks the bounds on parameters. Bounds should be absolute restrictions on possible parameter values, whereas constraints should be restrictions on parameter values relative to each other.

        Parameters
        ----------
        params: list[float]
            The list of parameter values for the free parameters of the model. The order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        
        Returns
        -------
        float
            A violation score, which is negative if any of the bounds are violated and non-negative otherwise. The violation score is the largest negative value from all the bounds.
        """

        violation_score = 0
        if not self.parameter_handler.has_been_fixed:
            for param_name, param_object in self.model_base_params.items():
                violation = self.get_param_value(param_name=param_name,
                                                params=params) - param_object.bounds[0]
                if violation < violation_score:
                    self.logger.debug(
                        f'Lower bound for parameter {param_name} is {param_object.bounds[0]}. '
                        f'Out of bounds by: {-violation}.')
                    violation_score = violation
                violation = param_object.bounds[1] - self.get_param_value(param_name=param_name,
                                                                        params=params)
                if violation < violation_score:
                    self.logger.debug(
                        f'Upper bound for parameter {param_name} is {param_object.bounds[1]}. '
                        f'Out of bounds by: {-violation}.')
                    violation_score = violation
        else:
            if len(params) != len(self.model_base_params):
                full_params = self.insert_params(params.copy(), [0 for _ in self.params_fixed_by_ancestry])
            else:
                full_params = params
 
            for param_name, param_object in self.model_base_params.items():
                violation = self.get_param_value(param_name=param_name,
                                                params=full_params) - param_object.bounds[0]
                if violation < violation_score:
                    self.logger.debug(f'Lower bound for parameter {param_name} is {param_object.bounds[0]}. Current value is {self.get_param_value(param_name=param_name, params=full_params)}.')
                    violation_score = violation
                violation = param_object.bounds[1] - self.get_param_value(param_name=param_name,
                                                                        params=full_params)
                if violation < violation_score:
                    self.logger.debug(f'Upper bound for parameter {param_name} is {param_object.bounds[1]}. Current value is {self.get_param_value(param_name=param_name, params=full_params)}.')
                    violation_score = violation
        return violation_score

    @staticmethod
    def parse_proportions(ancestor_names: list[str], proportions: list[str]) -> tuple[dict[str:str], str]:
        """
        Parses the ancestry proportions used in a founding event into a dict of parametrized source populations and a remainder population.

        Parameters
        ----------
        ancestor_names: list[str]
            A list of the names of the ancestor populations in the founding event. The order of the names should correspond to the order of the proportions in ``proportions``.
        proportions: list
            A list of the proportions of ancestry contributed by each ancestor population in the founding event. The order of the proportions should correspond to the order of the names in ``ancestor_names``. One of the proportions should be a string of the form "1-[the other proportions]", which indicates that the proportion for that ancestor population is equal to 1 minus the sum of the other proportions.
        
        Returns
        -------
        tuple[dict[str:str], str]
            A tuple of the form (``source_populations``, ``remainder_population``), where ``source_populations`` is a dict mapping the names of the ancestor populations with parametrized proportions to their corresponding proportion parameters, and ``remainder_population`` is the name of the ancestor population whose proportion is equal to 1 minus the sum of the other proportions.
        """
        #TODO: Add support for constants in proportions.
        #NOTE: May later be folded into the add_founder_event() method.

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
                                      'if two of the proportions are "a" and "b", the other must be "1-a-b."')

        # Check if the "1-" expression correctly contains all the other parameters.
        if not all(p1 == p2 for p1, p2 in
                   zip(source_populations.values(), remainder_proportion_string.split('-')[1:])):
            raise ValueError(
                'The given proportions are not guaranteed to sum to 1.\n'
                'When using parametrized founding proportions, a population must be specified '
                'whose proportion takes the form "1-[the other proportions]".\n'
                'For example, in a three-population founder event, if two of the proportions are "a" and "b",'
                ' the other must be "1-a-b."')

        return source_populations, remainder_population
    
    def _list_parameters(self):
        """
        Prints the parameters of the model and their types.
        """
        for param_name, param_info in self.model_base_params.items():
            print(f"{param_name}: {param_info.type}")
        return

    def set_up_fixed_parameters(self, params_to_fix_by_ancestry: list[str], 
                                proportions: dict[str: list[float]], params_to_fix_by_value:dict[str:float]={}):
        """
        Tells the model to calculate certain rate parameters based on the known ancestry proportions of the sample populations. Proportions are given as a dict with keys corresponding to the sample populations.

        Parameters
        ----------
        params_to_fix_by_ancestry: list[str]
            A list of the names of the parameters to be fixed by ancestry proportions.
        proportions: dict[str: list[float]]
            A dict mapping sample population names to their corresponding ancestry proportions, which are used to fix the parameters in ``params_to_fix_by_ancestry``.
        params_to_fix_by_value: dict[str:float]
            A dict mapping parameter names to their corresponding values, which are used to fix parameters by user-defined values. These parameters cannot be optimized and are not computed from ancestry proportions.
        """

        self.parameter_handler.set_up_fixed_parameters(demography = self,
                                                    params_to_fix_by_ancestry=params_to_fix_by_ancestry, 
                                                    proportions=proportions,
                                                    user_params_to_fix_by_value=params_to_fix_by_value)
                                  
    @abstractmethod
    def get_random_parameters():
        pass

    @abstractmethod
    def get_migration_matrices(self, params: list[float]) -> dict[str, np.ndarray]:
        pass

    @abstractmethod
    def add_pulse_migration(self, source_population, rate_param, time_param):
        pass

    @abstractmethod
    def add_continuous_migration(self, source_population, rate_param, start_param, end_param):
        pass

class FixedParametersHandler:
    """
    A class that handles the fixing of parameters based on ancestry proportions. This class is used by :class:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography` to fix parameters based on known ancestry proportions.

    Attributes
    ----------
    logger: logging.Logger
        The logger.
    params_not_fixed_by_ancestry: list[str]
        A list of the names of the parameters that are not fixed by ancestry proportions.
    params_fixed_by_ancestry: dict[str, str]
            A dict of the names of the parameters that are fixed by ancestry proportions and their corresponding values.
    known_ancestry_proportions: dict[str, list[float]] | None
        A dict mapping sample population names to their corresponding ancestry proportions.
    reduced_constraints: list[dict]
            A list of constraints that have been reduced from the original constraints after fixing parameters by ancestry proportions. These constraints are of the same form as the original constraints, but they only involve the free parameters that are being optimized after fixing parameters by ancestry proportions.
    user_params_fixed_by_value: dict[str, float]
            A dict of the names of the parameters that are fixed by user-defined values and their corresponding values.
    demography: BaseParametrizedDemography
        The demography object that this FixedParametersHandler is associated with. 
    to_physical_params_functions: dict[str, Callable]
        A dict mapping parameter types to functions that convert parameters from optimization units to physical units.
    to_optimizer_params_functions: dict[str, Callable]
        A dict mapping parameter types to functions that convert parameters from physical units to optimization units.
    """


    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.params_not_fixed_by_ancestry = []
        self.params_fixed_by_ancestry = {}
        self.known_ancestry_proportions: dict[str, list[float]] = None
        self.reduced_constraints =[]
        self.user_params_fixed_by_value = {}
        self.demography = None
        self.to_physical_params_functions = {}
        self.to_optimizer_params_functions = {}

    @property
    def has_been_fixed(self):
        """
        Whether any parameters have been fixed by ancestry proportions.

        Returns
        -------
        bool
            True if any parameters have been fixed by ancestry proportions, False otherwise.
        """
        return self.known_ancestry_proportions is not None


    def order_fixed_param_dict(self, fixed_params_dict: dict[str: float]):
        """
        Orders the given dict of fixed parameters by the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.

        Parameters
        ----------
        fixed_params_dict : dict[str, float]
            A dictionary mapping parameter names to their fixed values.

        Returns
        -------
        dict[str, float]
            A dictionary mapping parameter names to their fixed values, ordered by the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        """
    
        return {param_name: fixed_params_dict[param_name] for param_name in self.demography.model_base_params if
                                         param_name in fixed_params_dict}
    
    def get_sorted_indices(self, param_list: list[str]):
        """
        Computes the list of indices as they appear in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        
        Parameters
        ----------
        param_list: list[str]
            A list of parameter names for which to get the indices.
        
        Returns
        -------
        np.ndarray
            An array of indices corresponding to the positions of the parameters in ``param_list`` as they appear in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        """ 
        return np.array([index for index, param_name in enumerate(self.demography.model_base_params) if
                                        param_name in param_list], dtype=int)

    def convert_to_physical_params(self, optimizer_params: list[float]):
        """
        Converts optimizer parameters from optimization units to physical units.
        
        Parameters
        ----------
        optimizer_params: list[float]
            A list of parameter values for the free parameters of the model in optimization units. The order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        
        Returns
        -------
        np.ndarray
            An array of parameter values for the free parameters of the model in physical units, in the same order as in ``optimizer_params``.
        """
        optimizer_params = np.asarray(optimizer_params) 
        assert np.asarray(optimizer_params).ndim == 1
        converted_params = optimizer_params.copy()

        for index in range(len(optimizer_params)):
            param_name = list(self.demography.model_base_params.keys())[index]
            param_type = self.demography.model_base_params[param_name].type

            if param_type in self.to_physical_params_functions.keys():
                converted_params[index] = self.to_physical_params_functions[param_type](optimizer_params[index])
            if param_type == ParamType.TIME:
                if converted_params[index] > 15:
                    print(f'Time parameter {param_name} is too large after conversion to physical units: {converted_params[index]}. Check scaling functions.')

        return converted_params


    def convert_to_optimizer_params(self, physical_params: list[float]):
        """
        Converts parameters from optimization units to physical units.
        
        Parameters
        ----------
        physical_params: list[float]
            A list of parameter values for the free parameters of the model in physical units. The order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        
        Returns
        -------
        np.ndarray
            An array of parameter values for the free parameters of the model in optimization units, in the same order as in ``physical_params``.
        """
        assert np.asarray(physical_params).ndim == 1
        converted_params = physical_params.copy()

        for index in range(len(physical_params)):
            param_name = list(self.demography.model_base_params.keys())[index]
            param_type = self.demography.model_base_params[param_name].type
            if param_type in self.to_optimizer_params_functions.keys():
                converted_params[index] = self.to_optimizer_params_functions[param_type](physical_params[index])

        return converted_params


    def set_up_fixed_parameters(self, demography: BaseParametrizedDemography, params_to_fix_by_ancestry: list[str],
                                    proportions: dict[str: list[float]], user_params_to_fix_by_value:dict[str:float]={}):
        """
        Tells the model to calculate certain rate parameters based on the known ancestry proportions of the sample populations, or to fix them by value. 
        
        Parameters
        ----------
        demography: BaseParametrizedDemography
            The demography object that this FixedParametersHandler is associated with.
        params_to_fix_by_ancestry: list[str]
            A list of the names of the parameters to be fixed by ancestry proportions.
        proportions: dict[str: list[float]]
            A dict mapping sample population names to their corresponding ancestry proportions, which are used to fix the parameters in ``params_to_fix_by_ancestry``.
        user_params_to_fix_by_value: dict[str:float]
            A dict mapping parameter names to their corresponding values, which are used to fix parameters by user-defined values. These parameters cannot be optimized and are not computed from ancestry proportions.       
        """

        self.demography = demography
        self.user_params_fixed_by_value = self.order_fixed_param_dict(user_params_to_fix_by_value)
        self.current_fixed_parameters = self.user_params_fixed_by_value.copy()
        self.params_fixed_by_value_indices = self.get_sorted_indices(user_params_to_fix_by_value.keys())  
        self.params_fixed_by_ancestry_indices = self.get_sorted_indices(params_to_fix_by_ancestry)
        self.params_fixed_by_values_values =list(self.user_params_fixed_by_value.values())
        
        self.free_parameters_indices = [index for index, param_name in enumerate(self.demography.model_base_params) if
                                         param_name not in user_params_to_fix_by_value and param_name not in params_to_fix_by_ancestry]

        self.demography.fixed_parameter_values = {user_params_to_fix_by_value[param_name] for param_name in demography.model_base_params if
                                         param_name in user_params_to_fix_by_value} # Store value in the demography object for easy access when computing parameters fixed by ancestry.  
        
        if len(params_to_fix_by_ancestry)!= 0 and not (self.demography.proportions_from_matrices_return_keys() == proportions.keys()):
            raise KeyError(
                "The keys of the provided sample proportions do not match proportions from matrices():"
                f"\nExpected keys: {demography.proportions_from_matrices_return_keys()}."
                f"\nProvided keys: {proportions.keys()}."
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
            
        if len(params_to_fix_by_ancestry) not in [0, sum(len(prop)-1 for prop in proportions.values())]:
            raise ValueError(
                    f'Number of parameters to fix is incorrect.'
                    f'Each population of interest can have N-1 proportions fixed'
                    f'Where N is the number of ancestral sources for that population'
                )
        
        # Use a dict to maintain order. Also looping over demography.model_base_params rather than params_to_fix to maintain order.
        self.params_fixed_by_ancestry = {param_name: '' for param_name in demography.model_base_params if
                                         param_name in params_to_fix_by_ancestry}
        
        # Exclude the last set of proportions because they are redundant (proportions must sum to 1).
        self.known_ancestry_proportions = {key:prop[:-1] for key, prop in proportions.items()}
        
        # Keep the constraints that involve any of the fixed parameters. Not used yet.
        self.reduced_constraints = [constraint for constraint in self.demography.constraints if any(
            param_name in self.params_fixed_by_ancestry for param_name in constraint['param_subset'])]
    
    def extend_parameters(self, free_parameters: np.ndarray, units: str | None = None):
        """
        Takes in the free parameters (those seen by the optimizer) and extends them to include the 
        full parameter list by computing the parameters fixed by ancestry after adding parameters fixed by value.

        Parameters
        ----------
        free_parameters: np.ndarray
            An array of parameter values for the free parameters of the model that are being optimized by the primary optimizer. The order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params` that are not fixed by ancestry proportions or by user-defined values.
        units: str | None
            The units of the input parameters. If "phys", the input parameters are in physical units. If "opt", the input parameters are in optimization units. If None, defaults to "phys". The units of the output parameters are the same as the input parameters.
        
        Returns
        -------
        np.ndarray
            An array of parameter values for all the free parameters of the model, including those fixed by ancestry proportions and by user-defined values, in the same order as in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        """

        units = units if units is not None else "phys"      
        full_parameters = np.zeros(len(self.demography.model_base_params), dtype=float)
        full_parameters[self.free_parameters_indices] = free_parameters
        full_parameters[self.params_fixed_by_value_indices] = list(self.params_fixed_by_values_values)
        
        try:
            return self.compute_params_fixed_by_ancestry(params=full_parameters,
                                                        units = units) 
        except ValueError as e:
            logger.warning(f"Could not extend parameters at {full_parameters}, defaulting to zeros for unknown params.")
            return full_parameters

    def indices_to_labels(self, indices: list[int]):
        """
        Takes in a list of indices and returns the corresponding parameter names.
        
        Parameters
        ----------
        indices: list[int]
            A list of indices corresponding to the positions of parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        
        Returns
        -------
        list[str]
            A list of parameter names corresponding to the given indices.
        """
        keys = list(self.demography.model_base_params.keys())
        return [keys[i] for i in indices]
    

    def reduce_parameters(self, full_parameters: list[float]):
        """
        Takes in the full set of parameters for the demography and reduces them to only the free parameters.
        
        Parameters
        ----------
        full_parameters: list[float]
            A list of parameter values for all the free parameters of the model, including those fixed by ancestry proportions and by user-defined values, in the same order as in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        
        Returns
        -------
        np.ndarray
            An array of parameter values for the free parameters of the model that are being optimized by the primary optimizer, in the same order as in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params` that are not fixed by ancestry proportions or by user-defined values.
        """
        return full_parameters[self.free_parameters_indices]

    def add_fixed_parameters(self, new_fixed_params: dict[str, float]):
        """
        Adds new fixed parameters by value to the current set of fixed parameters, and updates the indices of the free parameters accordingly. Checks that the new fixed parameters do not conflict with any existing fixed parameters.

        Parameters
        ----------
        new_fixed_params: dict[str, float]
            A dict mapping parameter names to their corresponding values, which are used to fix parameters by user-defined values. These parameters cannot be optimized and are not computed from ancestry proportions.
        """
        intersection = set(new_fixed_params.keys()).intersection(set(self.user_params_fixed_by_value.keys()))
        assert not intersection, f'Parameters {intersection} are already fixed by value.' 
        
        self.current_fixed_parameters.update(new_fixed_params)
        self.current_fixed_parameters = self.order_fixed_param_dict(self.current_fixed_parameters)
        self.params_fixed_by_value_indices = self.get_sorted_indices(self.current_fixed_parameters.keys())
        self.params_fixed_by_values_values =self.current_fixed_parameters.values()
        self.free_parameters_indices = [index for index, param_name in enumerate(self.demography.model_base_params) if
                                         param_name not in self.current_fixed_parameters and param_name not in self.params_fixed_by_ancestry]        
    
    def release_fixed_parameters(self, freed_params: list[str]):
        """
        Removes parameters from the current set of fixed parameters, and updates the indices of the free parameters accordingly.

        Parameters
        ----------
        freed_params: list[str]
            A list of parameter names corresponding to parameters that are currently fixed by value that should be released and made free to be optimized by the primary optimizer.
        """
        self.current_fixed_parameters = {param_name: value for param_name, value in self.current_fixed_parameters.items() if param_name not in freed_params}
        self.params_fixed_by_value_indices = self.get_sorted_indices(self.current_fixed_parameters.keys())
        self.params_fixed_by_values_values =self.current_fixed_parameters.values()
        self.free_parameters_indices = [index for index, param_name in enumerate(self.demography.model_base_params) if
                                         param_name not in self.current_fixed_parameters and param_name not in self.params_fixed_by_ancestry]
        

    def full_params_objective_func(self, parameters: list[float], units: str | None = None):
        """
        Returns the difference between computed and model ancestry proportions, as an array.
        
        Parameters
        ----------
        parameters: list[float]
            A list of parameter values for all the free parameters of the model, including those fixed by ancestry proportions and by user-defined values, in the same order as in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        units: str | None
            The units of the input parameters. Only implemented for physical parameters so the function will raise an error if ``units`` is not ``"phys"``.

        Returns
        -------
        np.ndarray
            An array of the differences between the computed ancestry proportions from the given parameters and the known ancestry proportions, for each sample population and each ancestor population with a fixed proportion, in the same order as in ``known_ancestry_proportions``.
        """
        
        assert units == "phys", "Not implemented for optimizer units."

        migration_matrices = self.demography.get_migration_matrices(parameters)
        found_props = self.demography.proportions_from_matrices(migration_matrices=migration_matrices)
        diff = np.array([found_props[ancestor][:-1] - self.known_ancestry_proportions[ancestor] 
                for ancestor in self.known_ancestry_proportions.keys()]).flatten()
        return diff  

    
    def compute_params_fixed_by_ancestry(self, params: list[float], known_ancestry_proportions: dict[str, np.ndarray] | None = None, units: str | None = None):
        """
        Compute the parameters fixed by ancestry proportions.

        Parameters
        ----------
        params : list[float]
            The input parameters for the demography model.
        known_ancestry_proportions : dict[str, np.ndarray], optional
            A dictionary mapping ancestor population names to their known ancestry proportions.
        units : str, optional
            The units of the input parameters. If "phys", the input parameters are in physical units. If "opt", the input parameters are in optimization units. If None, defaults to "phys". The units of the output parameters are the same as the input parameters.

        Returns
        -------
        list[float]
            The computed parameters for the fixed ancestry proportion model.
        """
        
        units = units if units is not None else "phys"
        if units == "opt":
            params_phys = self.convert_to_physical_params(optimizer_params=params)
            params_opt = params.copy()
        else: 
            assert units == "phys", "units must be 'phys' or 'opt'."
            params_phys = params.copy()
            params_opt = self.convert_to_optimizer_params(physical_params=params)

        if not self.has_been_fixed and len(params_phys) != len(self.demography.model_base_params): #TODO: discard the second condition?
            raise Exception("The demography has not been fixed yet.")
        if known_ancestry_proportions==None:
            known_ancestry_proportions=self.known_ancestry_proportions

        self.logger.debug(f'Params before fixed-ancestry solving: {params_phys}.')
        assert (len(params_phys) == len(self.demography.model_base_params)), f"Length of input parameters {len(params_phys)} does not match number of model parameters {len(self.demography.model_base_params)}."
     
        migration_matrices = self.demography.get_migration_matrices(params=params_phys)
        try:
            calculated_proportions = self.demography.proportions_from_matrices(migration_matrices=migration_matrices)  
            if np.all([np.allclose(calculated_proportions[sample_pop][:-1], known_ancestry_proportions[sample_pop])
                        for sample_pop in known_ancestry_proportions.keys()]):
                if units == "opt":
                    params_phys = self.convert_to_optimizer_params(physical_params=params_phys)
                return params_phys
        except ValueError: # Catches cases where the parameters produce invalid matrices
            pass


        def param_objective_func(params_to_solve: np.ndarray, large: float = 1e12): 
            """
            Computes the difference between observed and computed ancestry proportions as a function of ``params_to_solve``, in optimizer units. 
            Uses the physical parameters as a non-local variable.
             
            Parameters
            ----------
            params_to_solve: np.ndarray
                An array of parameter values for the parameters that are fixed by ancestry proportions, in optimization units. The order of the values corresponds to the order of the parameters in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params` that are fixed by ancestry proportions.
            large: float
                A large number to add to the objective function when the parameters produce invalid migration matrices or when the parameters fixed by ancestry proportions are out of bounds, to discourage the optimizer from exploring those parameter values.
            
            Returns
            -------
            np.ndarray
                An array of the differences between the computed ancestry proportions from the given parameters and the known ancestry proportions, for each sample population and each ancestor population with a fixed proportion, in the same order as in ``known_ancestry_proportions``.
            """
            nonlocal params_opt
            
            params_to_solve[np.isnan(params_to_solve)] = 0
            new_params_phys = self.convert_to_physical_params(optimizer_params=self.insert_solved_params(full_params=params_opt,
                                                                                        param_values_from_proportions=params_to_solve)) 
            try: value = self.full_params_objective_func(parameters=new_params_phys,
                                                        units = "phys") 
            except ValueError as e:
                self.logger.warning(f"Problem computing migration matrices with opt parameters {params_opt}, physical parameters {params_phys}.")
                return large
            
            bound = self.demography.check_bounds(params=start_params_phys_full) 
            if bound < 0: 
                return value + (1-bound)*large
            else:
                return value
                

        start_point = np.ones(len(self.params_fixed_by_ancestry)) * .1 # An arbitrary starting point in physical units.
        start_params_phys_full = self.insert_solved_params(full_params=params_phys,
                                                            param_values_from_proportions=start_point)
        assert(self.demography.check_bounds(params=start_params_phys_full) >=0), "Starting point for fixed parameter optimisation is not feasible." #TODO: Come up with a way of catching and repairing unfeasible starting points.
        start_point_optimizer_full = self.convert_to_optimizer_params(physical_params=start_params_phys_full)
        start_point_validated =  start_point_optimizer_full[self.params_fixed_by_ancestry_indices]
        
        try: 
            solved_params = scipy.optimize.fsolve(func=param_objective_func,
                                                x0=start_point_validated)
        except (ValueError, TypeError) as e:
            raise ValueError("Could not solve for parameters fixed by ancestry proportions.") from e

        error = np.linalg.norm(param_objective_func(solved_params))
        if not np.isclose(error, 0):
            self.logger.warning(f"Could not solve for parameters fixed by ancestry proportions. Final error: {error}, solved_params (physical) = {self.convert_to_physical_params(self.insert_solved_params(full_params=params_opt, param_values_from_proportions=solved_params))} this can happen when no sex bias parameter allows for the observed ancestry proportions.")
            
        if np.isnan(solved_params).any():
            print ("Could not solve for parameters fixed by ancestry proportions. Some parameters are NaN.")

        params_phys = self.convert_to_physical_params(self.insert_solved_params(full_params=self.convert_to_optimizer_params(physical_params=params_phys),
                                                                                param_values_from_proportions=solved_params))
        self.logger.debug(f'Params after solving with ancestry proportions: {params_phys}.')
         
        if units == "opt":
            return self.convert_to_optimizer_params(physical_params=params_phys)
        return params_phys


    def insert_solved_params(self, full_params: list[float], param_values_from_proportions: list[float]):
        """
        Merges the parameters solved by the primary optimizer with the parameters found from the known ancestry proportions
        into a single list of parameters in the correct order for the model.

        Parameters
        ----------
        full_params: list[float]
            A list of parameter values for all the free parameters of the model, including those fixed by ancestry proportions and by user-defined values, in the same order as in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        param_values_from_proportions: list[float]
            A list of parameter values for the parameters fixed by ancestry proportions, in the same order as in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params` for the parameters fixed by ancestry proportions.
        
        Returns
        -------
        np.ndarray
            An array of parameter values for all the free parameters of the model, including those fixed by ancestry proportions and by user-defined values, in the same order as in :py:attr:`~tracts.demography.base_parametrized_demography.BaseParametrizedDemography.model_base_params`.
        """
        assert self.params_fixed_by_ancestry_indices.dtype == int
        output_params = np.array(full_params, dtype=float, copy=True) 
        output_params[self.params_fixed_by_ancestry_indices] = np.array(param_values_from_proportions)
        
        return output_params


    def insert_fixed_params(self, model_base_params: dict[str, Parameter], params_to_optimize: list[float], fixed_params: list[float]):
        """
        Inserts the fixed parameter values into the list of parameters to be optimized.
        
        Parameters
        ----------
        model_base_params : dict[str, Parameter]
            A dictionary of all base parameters for the model.
        params_to_optimize : list[float]
            A list of values for the parameters fixed by neither values nor ancestry proportions.
        fixed_params : list[float]
            A list of parameter values for the parameters fixed by value.
        
        Returns
        -------
        list[float]
            A list of parameter values for all the non-computed parameters.
        """        
        #NOTE: This could be refactored with insert solved parameters with the parameters found from the known ancestry proportions.
        
        assert (len(params_to_optimize) + len(fixed_params)+len(self.params_fixed_by_ancestry) == len(model_base_params)), f"{len(params_to_optimize)} + {len(fixed_params)}+{len(self.params_fixed_by_ancestry)} == {len(model_base_params)} non-computed parameters: {model_base_params}."
        
        iter_params = iter(params_to_optimize)
        iter_fixed_params = iter(fixed_params)
        params_to_optimize = [next(iter_fixed_params) if (param_name in self.user_params_fixed_by_value) else next(iter_params)  
                      for param_name in model_base_params if param_name not in self.params_fixed_by_ancestry ]
        return params_to_optimize
        

    def check_for_unsolvable_proportions(self, demography: BaseParametrizedDemography):
        """
        Checks that the demography has an assignment of (full) parameters that results in the chosen proportions.

        Parameters
        ----------
        demography: BaseParametrizedDemography
            The demography object that this :class:`~tracts.base_parametrized_demography.FixedParametersHandler` is associated with. 
        """
        def objective_func(params):
            migration_matrices = demography.get_migration_matrices(params)
            diff = [prop[:-1] - self.known_ancestry_proportions[sample_pop]
                    for sample_pop, prop in demography.proportions_from_matrices(
                        migration_matrices=migration_matrices).items()]
            return np.linalg.norm(diff)
        
        result = scipy.optimize.minimize(objective_func, 
                                        x0=demography.get_random_parameters(),
                                        bounds=demography.parameter_bounds,
                                        constraints={'type': 'ineq', 'fun': demography.check_constraints})
        if not np.isclose(result.fun,0):
            raise ValueError(
                'The ancestry proportions in the sample are not achievable with the provided demographic model.')

    def check_for_improper_constraint(self, demography: BaseParametrizedDemography):
        """
        Checks that the choice of parameters to fix does not underconstrain or overconstrain any of the matrices.

        Parameters
        ----------
        demography: BaseParametrizedDemography
            The demography object that this :class:`~tracts.base_parametrized_demography.FixedParametersHandler` is associated with. 
        """
        starting_params = demography.get_random_parameters()
        target_matrices = demography.get_migration_matrices(starting_params)
        target_proportions = demography.proportions_from_matrices(target_matrices)
        # TODO: This function is unfinished.
