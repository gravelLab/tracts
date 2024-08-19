import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
                #path to tracts, may have to adjust if the file is moved
import tracts
import numpy
import scipy
import matplotlib.pyplot as plt
from pprint import pprint

'''
Tests for component methods of tracts core
'''

def test_PDT():
    migration_matrix = numpy.array([[0,0], [0,0],[0.1,0],[0,0],[0,0],[0,0],[0.2,0.8]])
    PTD = tracts.PhaseTypeDistribution(migration_matrix)
    assert numpy.isclose(numpy.linalg.det(PTD.equilibrium_distribution), 0) 
    assert numpy.isclose(numpy.linalg.norm(PTD.equilibrium_distribution), 1)
    assert numpy.allclose(numpy.dot(PTD.equilibrium_distribution, PTD.full_transition_matrix), 0)

def verify_PDT():
    '''
    Test that phase-type distributions gives the same result as demographic_model.expectperbin()
    ''' 

    bins = [0.1, 0.15561805, 0.21123611, 0.26685416, 0.32247222, 0.37809027,
            0.43370833, 0.48932638, 0.54494444, 0.60056249, 0.65618055, 0.7117986,
            0.76741666, 0.82303471, 0.87865277, 0.93427082, 0.98988888, 1.04550693,
            1.10112498, 1.15674304, 1.21236109, 1.26797915, 1.3235972, 1.37921526,
            1.43483331, 1.49045137, 1.54606942, 1.60168748, 1.65730553, 1.71292359,
            1.76854164, 1.8241597,  1.87977775, 1.93539581, 1.99101386, 2.04663191,
            2.10224997, 2.15786802, 2.21348608, 2.26910413, 2.32472219, 2.38034024,
            2.4359583, 2.49157635, 2.54719441, 2.60281246, 2.65843052, 2.71404857,
            2.76966663]
    Ls = [2.7809027354399998, 1.7928357829, 1.5953700000000002, 1.728997512, 1.27096679, 1.1710800000000001, 1.31433, 2.6344739744, 2.24544150046, 2.1288176, 2.0397778748, 1.9301017528000002, 1.8692348767000002, 1.7021899999999999, 1.6825700000000001]
    pops = [0,1]
    A = numpy.array([[0,0], [0,0],[0.1,0],[0,0],[0,0],[0,0],[0.2,0.8]])
    B = numpy.array([[0,0], [0,0], [0,0.5], [0.6,0.4]])
    C = numpy.array([[0,0], [0,0], [0,0.5], [0.2,0.2], [0.6,0.4]])
    D = numpy.array([[0,0], [0,0], [0,0.3], [1,0]])
    print('Migration Matrix')
    print(A)
    #compare_TpopTau(A)
    #print(A.nonzero()[1])
    #PTD_D_hist = viewPTD(D, 0, bins)
    #compare_models(A, bins, 1, 1)
    #compare_models_2(A, bins, [1], 1)
    #compare_models_4(A, bins, Ls[0], 0)
    #print(weird_factor(A, Ls, 0))
    models = ModelComparison(A)
    #models.compare_proportions()
    models.compare_models(bins, Ls[0], 0)
    #models.compare_models_2(bins, Ls[0], 0)
    #compare_models_5(A, bins, Ls, 0)
    

def view_PTD_CDFs(migration_matrix):
    PTD = tracts.PhaseTypeDistribution(migration_matrix)

class ModelComparison:
    '''
    A class for comparing PhaseType and demographic_model values
    '''
    def __init__(self, migration_matrix):
        self.migration_matrix = migration_matrix
        self.PTD = tracts.PhaseTypeDistribution(migration_matrix)
        self.demo = tracts.demographic_model(migration_matrix)

    def compare_proportions(self):
        print(f'Tracts proportions at t0: {self.demo.proportions[0]}')
        print(f'PTD proportions at t0: {self.PTD.t0_proportions}')
    
    def compare_models(self, bins, L, population_number):
        PTD_hist = self.PTD.tractlength_histogram_windowed(population_number, bins, L)
        demo_hist = self.demo.expectperbin([L],population_number,bins)
        print(f'\nTractlength histogram from PTD: \n{PTD_hist}')
        print(f'\nTractlength histogram from tracts: \n{numpy.array(demo_hist)}')

    def compare_models_2(self, bins, L, population_number):
        PTD_hist = self.PTD.tractlength_histogram_windowed(population_number, bins, L)
        demo_hist = self.demo.expectperbin([L],population_number,bins)
        print(sum(PTD_hist))
        print(sum(demo_hist))
        fig, ax = plt.subplots(1,2)
        L = round(L,3)
        ax[0].loglog(bins, demo_hist, label = 'demographic_model')
        ax[0].loglog(bins, PTD_hist, label = 'Phase-Type Distribution')
        ax[0].set_title(f"Log-Log Plot of Expected Number of Tracts Per Bin\n On a chromosome of length {L}")
        ax[0].set_xlabel("Bin Value")
        ax[0].set_ylabel("Expected number of tracts")
        ax[1].plot(demo_hist, PTD_hist)
        ax[1].set_title(f"Ratio of Phase-Type Distribution tracts to demographic_model tracts\n On a chromosome of length {L}")
        ax[1].set_xlabel("demographic_model")
        ax[1].set_ylabel("Phase-Type Distribution")
        ratio = round(sum(demo_hist)/sum(PTD_hist),3)
        Rstat = scipy.stats.pearsonr(PTD_hist,demo_hist)
        ax[1].text(0.05, 0.95, f'Slope: {ratio} \nR-value: {round(Rstat.statistic, 3)} \nP-Value: {round(Rstat.pvalue,3)}', transform=ax[1].transAxes, fontsize=14,
            verticalalignment='top')
        plt.show()

    def compare_models_5(self, bins, Ls, population_number):
        print(f'Chromosome lengths:\n {Ls}')
        Z = numpy.array([self.PTD.normalization_factor([L], self.PTD.transition_matrices[population_number], self.PTD.inverse_S0_list[population_number], self.PTD.alpha_list[population_number]) for L in Ls])
        
        ratios = numpy.array([numpy.sum(self.demo.expectperbin([L], population_number, bins))/numpy.sum(self.PTD.tractlength_histogram_windowed(population_number, bins, L)) for L in Ls])
        print(f'\nRatios of expectperbin (tracts) to histogram (PhaseType): \n{ratios}')
        print(f'\nRatios divided by Z: \n{numpy.divide(ratios, Z)}')
        weird_factors = self.weird_factors(Ls, population_number)
        print(f'\nScaling factor from tracts: \n{weird_factors} ')
        print(f'\nScaling factor over Z: \n{numpy.divide(weird_factors, Z)}')

    def compare_models_6(self, population_number, Ls):
        #print(f'Alpha * S0_inverse:\n {self.alpha_s0_inv(population_number)} ')
        print(f'Chromosome lengths: \n{numpy.array(Ls)}')
        print(f'\nScaling factor from tracts: \n{self.weird_factors(Ls, population_number)} ')
        Z_to_tracts_factor = 2*self.PTD.t0_proportions[population_number]/self.alpha_s0_inv(population_number)
        Z = numpy.array([self.PTD.normalization_factor([L], self.PTD.transition_matrices[population_number], self.PTD.inverse_S0_list[population_number], self.PTD.alpha_list[population_number]) for L in Ls])
        print(f'\n  Z * 2 * (proportions at t0) / (alpha*s0_inverse) * -1: \n{-Z*Z_to_tracts_factor}')
        print(f'\nTracts self.totSwitchDensity:\n {self.demo.totSwitchDens[population_number]}')
        print(f'\n2 * (proportions at t0) / (alpha*s0_inverse) * -1: \n{-Z_to_tracts_factor}')
        return
    
    def alpha_s0_inv(self, population_number):
        '''
        The product of alpha and S0_inv for the given population
        '''
        return numpy.dot(self.PTD.alpha_list[population_number], self.PTD.inverse_S0_list[population_number])

    def weird_factors(self, Ls, population_number):
        return numpy.array([L*self.demo.totSwitchDens[population_number] + 2. * self.demo.proportions[0, population_number] for L in Ls])


def compare_models(migration_matrix, bins, L, population_number):
    PTD = tracts.PhaseTypeDistribution(migration_matrix)
    demo = tracts.demographic_model(migration_matrix)
    PTD_hist = PTD.tractlength_histogram_windowed(population_number, bins, L)
    demo_hist = demo.expectperbin([L],population_number,bins)
    print(sum(PTD_hist))
    print(sum(demo_hist))
    fig, ax = plt.subplots(1,2)
    L = round(L,3)
    ax[0].loglog(bins, demo_hist, label = 'demographic_model')
    ax[0].loglog(bins, PTD_hist, label = 'Phase-Type Distribution')
    ax[0].set_title(f"Log-Log Plot of Expected Number of Tracts Per Bin\n On a chromosome of length {L}")
    ax[0].set_xlabel("Bin Value")
    ax[0].set_ylabel("Expected number of tracts")
    ax[1].plot(demo_hist, PTD_hist)
    ax[1].set_title(f"Ratio of Phase-Type Distribution tracts to demographic_model tracts\n On a chromosome of length {L}")
    ax[1].set_xlabel("demographic_model")
    ax[1].set_ylabel("Phase-Type Distribution")
    ratio = round(sum(demo_hist)/sum(PTD_hist),3)
    Rstat = scipy.stats.pearsonr(PTD_hist,demo_hist)
    ax[1].text(0.05, 0.95, f'Slope: {ratio} \nR-value: {round(Rstat.statistic, 3)} \nP-Value: {round(Rstat.pvalue,3)}', transform=ax[1].transAxes, fontsize=14,
        verticalalignment='top')
    plt.show()

def compare_models_3(migration_matrix, bins, L, population_number):
    PTD = tracts.PhaseTypeDistribution(migration_matrix)
    PTD_hist = PTD.tractlength_histogram_windowed(population_number, bins, L)
    PTD_CDF = PTD.tractlength_CDF_windowed(population_number, bins, L)
    print(numpy.array([numpy.sum(PTD_hist[:i]) for i in range(len(PTD_hist)+1)]))
    print(PTD_CDF-PTD_CDF[0])

def compare_models_4(migration_matrix, bins, L, population_number):
    PTD = tracts.PhaseTypeDistribution(migration_matrix)
    demo = tracts.demographic_model(migration_matrix)
    PTD_hist = PTD.tractlength_histogram_windowed(population_number, bins, L)
    demo_hist = per_bin_noscale(demo, bins, L, population_number)
    demo_hist2 = per_bin_noscale(demo, bins, L, population_number)
    print(f'Chromosome has length {L}.')
    print(f'Normalized tractlength distribution from tracts PDF using bin midpoints:\n{demo_hist}')
    print(f'Normalized tractlength distribution from PhaseType CDF:\n{PTD_hist}')
    print(f'Normalized tractlength distribution from tracts PDF using scipy integration:\n {demo_hist2}')

def compare_models_2(migration_matrix, bins, Ls, population_number):
    PTD = tracts.PhaseTypeDistribution(migration_matrix)
    demo = tracts.demographic_model(migration_matrix)
    print(f'Z from tracts: {demo.Z(Ls, population_number)}')
    print(f'Z from PhaseType Calculation: {numpy.array([PTD.normalization_factor([L], PTD.transition_matrices[population_number], PTD.inverse_S0_list[population_number], PTD.alpha_list[population_number]) for L in Ls])}')
    print(f'Z from PhaseType CDF: {numpy.array([PTD.normalization_factor_2([L], PTD.transition_matrices[population_number], PTD.inverse_S0_list[population_number], PTD.alpha_list[population_number])[0] for L in Ls])}')
    #print(scipy.stats.pearsonr(PTD_hist,demo_hist))

def per_bin_noscale(demographic_model: tracts.demographic_model, bins, L, population_number):
    PDF = lambda x: (demographic_model.inners(L, x, population_number) + demographic_model.outers(L, x, population_number))/demographic_model.Z(L, population_number)
    binval = lambda binNum: scipy.integrate.quad(PDF, bins[binNum], bins[binNum+1])
    lsval = [binval(binNum)[0] for binNum in range(len(bins)-1)]
    lsval.append(demographic_model.full(L, population_number)/demographic_model.Z(L, population_number))
    return numpy.array(lsval)

def per_bin_noscale_integral(demographic_model: tracts.demographic_model, bins, L, population_number):
    mid = lambda binNum: (bins[binNum+1] + bins[binNum])/2
    diff = lambda binNum: bins[binNum+1] - bins[binNum]
    PDF = lambda x: (demographic_model.inners(L, x, population_number) + demographic_model.outers(L, x, population_number))/demographic_model.Z(L, population_number)
    
    lsval = [PDF(mid(binNum))*diff(binNum) for binNum in range(len(bins)-1)]
    lsval.append(demographic_model.full(L, population_number)/demographic_model.Z(L, population_number))
    return numpy.array(lsval)

def compare_TpopTau(migration_matrix):
    demo = tracts.demographic_model(migration_matrix)
    PTD = tracts.PhaseTypeDistribution(migration_matrix)
    for tpopTau, result in demo.dicTpopTau.items():
        #print(tpopTau, PTD.get_TpopTau(tpopTau[0], tpopTau[1], tpopTau[2]), result)
        assert PTD.get_TpopTau(tpopTau[0], tpopTau[1], tpopTau[2]) == result
    print('TpopTau is equal for PTD and demographic_model. All assertions passed.')

def expectperbin_ratios(migration_matrix, bins, Ls, population_number):
    demo = tracts.demographic_model(migration_matrix)
    PTD = tracts.PhaseTypeDistribution(migration_matrix)
    demo_hist = lambda L: demo.expectperbin([L],population_number,bins)
    PTD_hist = lambda L: PTD.tractlength_histogram_windowed(population_number, bins, L)
    return numpy.array([numpy.sum(demo_hist(L))/numpy.sum(PTD_hist(L)) for L in Ls])

def weird_factor(migration_matrix, Ls, population_number):
    demo = tracts.demographic_model(migration_matrix)
    return numpy.array([L*demo.totSwitchDens[population_number] + 2. * demo.proportions[0, population_number] for L in Ls])

verify_PDT()