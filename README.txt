Tracts is a set of classes and definitions used to model migration histories based on ancestry tracts in admixed individuals. Time-dependent gene-flow from multiple populations can be modeled.



Examples contains sample hapmap data and scripts to analyze them, including two different gene flow models.
It also contains a 3-population model for 1000 genomes puerto Rican data

Installation: 

Copy all files and folders locally (See "Download zip") on the github repository page)

"tracts.py" is a python module. All its functions can be used from the python interpreter or ipython after it has been imported. It should work out-of-the-box once you have python, and numpy, pylab, scipy installed. 

 

If you are an academic, I recommend installing the Anaconda (https://store.continuum.io/cshop/academicanaconda) distribution. Make sure not to pay for it! Click Anaconda Academic License; it should be free for those with edu e-mail addresses."

input

Tracts input is a set bed-style file describing the local ancestry of segments along the genome.
 The file has 2 extra columns for the cM positions of the segments. There are two input files per individuals (for each haploid genome copy).  

chrom		begin		end			assignment	cmBegin	cmEnd
chr13		0			18110261	UNKNOWN	0.0			0.19
chr13		18110261	28539742	YRI			0.19		22.193
chr13		28539742	28540421	UNKNOWN	22.193		22.193
chr13		28540421	91255067	CEU		22.193		84.7013


driver file

To maintain maximum flexibility, the options and models in tracts are set up in a driver file and a "model" file. Examples of both are provided in the distribution; these examples are the best starting points for the first-time. Tracts can be used interactively--when using the (i)python console, it is easy to examine and plot the different variables.    


Reading the tracts output:

The exemple files produce 5 output files, e.g.

boot0_-252.11_bins	boot0_-252.11_liks	boot0_-252.11_ord	boot0_-252.11_pred
boot0_-252.11_dat	boot0_-252.11_mig	boot0_-252.11_pars


boot0 means that this is bootstrap iteration 0, which in the convention used here means the fit with the real data
-252.11 is the likelihood of the best-fit model

_bins contains the bins used in the discretization
_dat are the observed counts in each bins
_pred are the predicted counts in each bin, according to the model
_mig is the inferred migration matrix, with the most recent generation at the top, and one column per migrant population
_pars are the optimal parameters
_liks contains the likelihoods in the model parameter space in the output format of scipy.optimizes' "brute" function: the first number is the best likelihood, the top matrices define the grid of parameters usedin the search, and the last matrix defines the likelihood at all grid points. see http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html






Setting up a demographic model:

The space of possible incoming migration matrices is quite large; if we have p migrant populations over g generations, there can be n*g different migration rates. To simplify this, we introduce simplified parametrized models that describe the full migration matrix in terms of a few parameters. These models may, for example,  involve a discrete number of admixture pulses, or periods of constant migrations rate. The user has full flexibility in defining these models; in python, one needs to write a function that takes parameters as an imput (such as the time of the onset of migration, migration rate p), and returns a migration matrix.

Here is the simplest example of such a function, implementing a single pulse of migration:

def pp((init_Eu,tstart)):       
        
    	"""a simple model in which populations Eu and AFR arrive discretely at first generation. If a time is not integer, the migration is divided between neighboring times proportional to the non-integer time fraction. 
        """
        tstart*=100 #the time is scaled by a factor 100 in this model to ease optimization with some routines that expect all parameters to have the same scale
         
        
        if  tstart<0:
                #time shouldn't be negative: that should be caught by constraint function. Return empty matrix
                gen=int(numpy.ceil(max(tstart,0)))+1
                mig=numpy.zeros((gen+1,2))
                return mig
                        
        gen=int(numpy.ceil(tstart))+1 #number of generations in the migration matrix
        frac=gen-tstart-1 			  #how close we are to the integer approximation 
        mig=numpy.zeros((gen+1,2))	  #placeholder migration matrix
        
        initNat=1-init_Eu			  #initial migration rates must sum up to one. 
        
        #replace a fraction at second generation to ensure a continuous model distribution with generation      
        mig[-1,:]=numpy.array([init_Eu,initNat])
        mig[-2,:]=frac*numpy.array([init_Eu,initNat])
        
        return mig


Some parameter values are inconsistent: times must be positive, and proportions of migrants must be between 0 and 1. We define an auxiliary function that verifies whether these conditions are met
It returns a number that is nonnegative if constraints are satisfied, and gets increasingly negative when they are more strongly violated.


def outofbounds_pp(params):
        #constraint function evaluating below zero when constraints not satisfied
        ret=1 #initialize the return variable to a positive value.
        (init_Eu,tstart)=params
        
        ret=min(1,1-init_Eu) #migration proportion must be between 0 and 1
        ret=min(ret,init_Eu)
        
        
        
        #generate the migration matrix and test for possible issues
        func=pp #specify the model
        mig=func(params) #get the migration matrix
        totmig=mig.sum(axis=1) #calculate the migration rate per generation
        
        ret=min(ret,-abs(totmig[-1]-1)+1e-8) #first generation migration must sum up to 1 
        ret=min(ret,-totmig[0],-totmig[1]) #no migrations are allowed in the first two generations
        
        ret=min(ret,10*min(1-totmig),10*min(totmig)) #migration at any given generation cannot be greater than 1
        
        
        
        ret=min(ret,tstart-.02) #start time must be at least two generations ago
                  
        return ret
        

The population is founded when two populations meet; at the first generation, we consider all individuals in the population as “migrants”, so the sum of migration frequencies at the first generation must sum up to one. If it isn’t, tracts will complain. 

Importantly, the optimizers in tracts assume that all parameters are continuous, but the underlying markov model uses discrete generations. 
When a time falls between two integers, the migrants are distributed across the neighboring integers, in such a way that the migration matrix changes “continuously”, in the sense that expected number of migrants 
. Continuous change is important, because likelihood optimizers can really struggle if the model is discontinuous in parameter space. 





See the example files for example usage. If something isn't clear, please let me know by filing an "new issue", or emailing me. 
 

  
  
