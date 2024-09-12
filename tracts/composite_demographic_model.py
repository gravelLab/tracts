from tracts.demographic_model import DemographicModel
import numpy as np
from scipy.special import gammaln


class CompositeDemographicModel:
    """ The class of demographic models that account for variance in the number
        of ancestors of individuals of the underlying population.

        Specifically, this is the demographic model constructed by the
        "multifracs" family of optimization routines.

        The expected tract counts per bin in the composite demographic model is
        simply a component-wise sum of the expected tract counts per bin across
        the component demographic models.

        The log-likelihood of the composite demographic model is the computed
        based on the combined expected tract counts per bin.
    """

    def __init__(self, model_function, parameters, proportions_list):
        """ Construct a composite demographic model, in which we consider split
            groups of individuals.

            Arguments;
                model_function (callable):
                    A function that produces a migration matrix given some
                    model parameters and fixed ancestry proportions.
                parameters:
                    The parameters given to the model function when the
                    component demographic models are built.
                proportions_list:
                    The lists of ancestry proportions used to construc each
                    component demographic model.
        """
        self.model_function = model_function
        self.parameters = parameters
        self.proportions_list = proportions_list

        # build the component models
        self.models = [DemographicModel(model_function(parameters, props)) for props in proportions_list]

        self.npops = self.models[0].npops

    def loglik(self, bins, Ls, data_list, nsamp_list, cutoff=0):
        """ Evaluate the log-likelihood of the composite demographic model.

            To compute the log-likelihood, we combine the expected count of
            tracts per bin in each of the component demographic models into the
            composite expected counts per bin. The expected counts per bin are
            compared with the sum across subgroups of the actual counts per

            likelihoods of the component demographic models.

            See demographic_model.loglik for more information about the
            specifics of the log-likelihood calculation.
        """
        # maxlen = max(Ls)
        data = sum(np.array(d) for d in data_list)

        s = 0
        # TODO: data and expects is an integers here, how can we index them?
        for i in range(self.npops):
            expects = self.expectperbin(Ls, i, bins, nsamp_list=nsamp_list)
            for j in range(cutoff, len(bins) - 1):
                dat = data[i][j]
                s += -expects[j] + dat * np.log(expects[j]) - gammaln(dat + 1.)

        return s

    def expectperbin(self, Ls, pop, bins, nsamp_list=None):
        """ A wrapper for demographic_model.expectperbin that yields a
            component-wise sum of the counts per bin in the underlying
            demographic models.
            Since the counts given by the demographic_model.expectperbin are
            normalized, performing a simple sum of the counts is not
            particularly meaningful; it throws away some of the structure
            that we have gained by using a composite model.
            Hence, the nsamp_list parameter allows for specifying the
            count of individuals in each of the groups represented by this
            composite_demographic_model, which is then used to rescale the
            counts reported by the expectperbin of the component demographic
            models.
        """
        if nsamp_list is None:
            nsamp_list = [1 for _ in range(len(self.proportions_list[0]))]

        return sum(nsamp * np.array(mod.expectperbin(Ls, pop, bins))
                   for nsamp, mod in zip(nsamp_list, self.models))

    def migs(self):
        """ Get the list of migration matrices of the component demographic
            models.
            This method merely projects the "mig" attribute from the component
            models.
        """
        return [m.migration_matrix for m in self.models]
