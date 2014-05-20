#!/usr/bin/env python

import sys
sys.path.append("../")
import tracts
import pp,pp_px
import numpy,pylab



directory="./G10/"


#number of short tract bins not used in inference.

cutoff=2

#number of repetitions for each model (to ensure convergence of optimization)
rep_pp=2
rep_pp_px=2

#only trio individuals
names=[ "NA19700","NA19701", "NA19704" , "NA19703", "NA19819" , "NA19818","NA19835" , "NA19834","NA19901" , "NA19900" ,"NA19909" , "NA19908","NA19917" , "NA19916" ,"NA19713" , "NA19982","NA20127" , "NA20126","NA20357" , "NA20356"]



chroms=['%d' % (i,) for i in range(1,23)]

#load the population
pop=tracts.population(names=names,fname=(directory,"",".bed"),selectchrom=chroms)
(bins, data)=pop.get_global_tractlengths(npts=50)


#choose order of populations and sort data accordingly
labels=['EUR','AFR']
data=[data[poplab] for poplab in labels]

#we're fixing the global ancestry proportions, so we only need one parameter
startparams=numpy.array([ 0.0683211  ]) # (initial admixture time). Times are measured in units of hundred generations (i.e., multiply the number by 100 to get the time in generations). The reason is that some python optimizers do a poor job when the parameters (time and ancestry proportions) are of different magnitudes. 
#you can also look at the "_mig" output file for a generation-by-generation breakdown of the migration rates.

Ls=pop.Ls
nind=pop.nind


#calculate the proportion of ancestry in each individual
bypopfrac=[[] for i in range(len(labels))]
for ind in pop.indivs:
	#a list of tracts with labels and names
	tractslst=ind.applychrom(tracts.chrom.tractlengths)
	#a flattened list of tracts with labels and names
	flattracts=[numpy.sum([item[1] for chromo in tractslst for sublist in chromo for item in sublist if item[0]==label]) for label in labels]
	for i in range(len(labels)):
		bypopfrac[i].append(flattracts[i]/ numpy.sum(flattracts))

props=map(numpy.mean,bypopfrac)


#we compare two models; single pulse versus two European pulses.
func=pp.pp_fix
bound=pp.outofbounds_pp_fix
func2=pp_px.pp_px_fix
bound2=pp_px.outofbounds_pp_px_fix
#((tstart,t2,nuEu_prop)) start time, time of second migration, proportion at second migration (proportion at first migration fixed
#by total ancestry proportion). times measured in units of 100 generations (see above)
#give two different starting conditions, with one starting near the single-pulse model
startparams2=numpy.array([  0.107152   ,  0.0438957  ,  0.051725  ])
startparams2p=numpy.array([  0.07152   ,  0.03  ,  1e-8  ])

optmod=tracts.demographic_model(func(startparams,props))




def randomize(arr,scale=2):
	#takes an array and multiplies every element by a factor between 0 and 2, uniformly. caps at 1.
	return map(lambda i: min(i,1),scale*numpy.random.random(arr.shape)*arr)

liks_orig_pp=[]
maxlik=-1e18
startrand=startparams
for i in range(rep_pp):
	xopt=tracts.optimize_cob_fracs2(startrand,bins,Ls,data,nind,func,props,outofbounds_fun=bound,cutoff=cutoff,epsilon=1e-2)
	#optimize_cob_fracs2 takes one additional parameter: the proportion of each ancestry that will be used to fix the parameters. 
	optmodlocal=tracts.demographic_model(func(xopt,props))
	loclik=optmod.loglik(bins,Ls,data,nind,cutoff=cutoff)
	if loclik>maxlik:
		optmod=optmodlocal
		optpars=xopt
	liks_orig_pp.append(loclik)
	
	startrand=randomize(startparams)
	
	
print "likelihoods found: " ,	liks_orig_pp

liks_orig_pp_px=[]
startrand2=startparams2
maxlik2=-1e18



for i in range(0,rep_pp_px):
	xopt2=tracts.optimize_cob_fracs2(startrand2,bins,Ls,data,nind,func2,props,outofbounds_fun=bound2,cutoff=cutoff,epsilon=1e-2)
	try:
		optmod2loc=tracts.demographic_model(func2(xopt2,props))
		loclik=optmod2loc.loglik(bins,Ls,data,nind,cutoff=cutoff)
		if loclik>maxlik2:
			optmod2=optmod2loc
			optpars=xopt2
	except:
		print "convergence error"
		loclik=-1e8
	liks_orig_pp_px.append(loclik)
	startrand2=randomize(startparams2)

lik1=optmod.loglik(bins,Ls,data,nind,cutoff=cutoff)
lik2=optmod2.loglik(bins,Ls,data,nind,cutoff=cutoff)

##################
#some plotting functions using python


pylab.figure(1)

colordict={"AFR":'blue',"EUR":'red',"UNKNOWN":'gray'}
plotbins = bins.copy()

for i in range(len(bins)-1):
	plotbins[i]=(bins[i]+bins[i+1])/2

for popnum in range(len(data)):
	poplab=labels[popnum]
	modexp=nind*numpy.array(optmod.expectperbin(Ls,popnum,bins))
	pylab.semilogy(plotbins,modexp, color=colordict[poplab])
	pylab.semilogy(plotbins,data[popnum],'o',color=colordict[poplab],label=poplab)

pylab.xlabel("tract length (Morgans)")
pylab.ylabel("counts")
pylab.legend()
pylab.axis([0,3,.1,10000])

pylab.figure(2)


for popnum in range(len(data)):
	poplab=labels[popnum]
	modexp2=nind*numpy.array(optmod2.expectperbin(Ls,popnum,bins))
	pylab.semilogy(plotbins,modexp2, color=colordict[poplab])
	pylab.semilogy(plotbins,data[popnum],'o',color=colordict[poplab],label=poplab)

pylab.xlabel("tract length (Morgans)")
pylab.ylabel("counts")
pylab.legend()
pylab.axis([0,3,.1,10000])

pylab.show()
print "optimal values found:" ,lik1,lik2
######################
#Save the data to file for external plotting, model 1

outdir="./out"
 
fbins=open(outdir+"_bins",'w')
fbins.write("\t".join(map(str,bins)))
fbins.close()

fdat=open(outdir+"_dat",'w')
for popnum in range(len(data)):

	fdat.write("\t".join(map(str,data[popnum]))+"\n")


fdat.close()

fmig=open(outdir+"_mig",'w')

for line in optmod.mig:
	fmig.write("\t".join(map(str,line))+"\n")
fmig.close()
fpred=open(outdir+"_pred",'w')

for popnum in range(len(data)):
	fpred.write("\t".join(map(str,pop.nind*numpy.array(optmod.expectperbin(Ls,popnum,bins))))+"\n")
fpred.close()

#################
#The first two files will be identical across models. We save an extra copy to facilitate the plotting.

outdir="./out2"
 
fbins=open(outdir+"_bins",'w')
fbins.write("\t".join(map(str,bins)))
fbins.close()

fdat=open(outdir+"_dat",'w')
for popnum in range(len(data)):

	fdat.write("\t".join(map(str,data[popnum]))+"\n")


fdat.close()

fmig2=open(outdir+"_mig",'w')

for line in optmod2.mig:
	fmig2.write("\t".join(map(str,line))+"\n")
fmig2.close()
fpred2=open(outdir+"_pred",'w')

for popnum in range(len(data)):
	fpred2.write("\t".join(map(str,pop.nind*numpy.array(optmod2.expectperbin(Ls,popnum,bins))))+"\n")
fpred2.close()






