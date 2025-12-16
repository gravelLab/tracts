Created a list of all individuals in the local ancestry calls downloaded from 1000G. Downloaded 1000G sex variables from 

integrated_call_samples_v3.20130502.ALL.panel), here: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/


Check for males
grep MXL integrated_call_samples_v3.20130502.ALL.panel | grep -v "female" > MXL/MXL_males.txt
grep MXL integrated_call_samples_v3.20130502.ALL.panel | grep "female" > MXL/MXL_females.txt

Get all individuals

in bed file directory, 

printf "%s\n" *.bed   | sed -E 's/_(A|B)_.*//'   | sort -u > individuals.txt

Checked which individuals were not included in the panel

awk '
NR==FNR { seen[$1]=1; next }
!($1 in seen) { print $1 }
' integrated_call_samples_v3.20130502.ALL.panel ./MXL/TrioPhased/individuals.txt 
NA19660
NA19675
NA19685


These are all relateds, according to 	20140625_related_individuals.txt from  https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/

awk '
NR==FNR { seen[$1]=1; next }
($1 in seen) { print $1 }
' 20140625_related_individuals.txt ./MXL/TrioPhased/individuals.txt
NA19660
NA19675
NA19685

So these individuals are included in local ancestry, but will not be independent. We need to remove them from the call list. 

awk '
NR==FNR { seen[$1]=1; next }
($1 in seen) { print $1 }
' 20140625_related_individuals.txt ./MXL/TrioPhased/individuals.txt
NA19660
NA19675
NA19685


awk '
NR==FNR { seen[$1]=1; next }
!($1 in seen) { print $1 }
' 20140625_related_individuals.txt ./MXL/TrioPhased/individuals.txt > ./MXL/TrioPhased/individuals_unrelated.txt

Output as csv to copy in driver file

awk '
NR==FNR { seen[$1]=1; next }
!($1 in seen) { printf "\"%s\",", $1}
' 20140625_related_individuals.txt ./MXL/TrioPhased/individuals.txt

Output males in similar fashion

awk '
NR==FNR { seen[$1]=1; next }
($1 in seen) { printf "\"%s\",", $1}
' ./MXL/MXL_males.txt ./MXL/TrioPhased/individuals_unrelated.txt
"NA19649","NA19652","NA19655","NA19658","NA19661","NA19664","NA19670","NA19676","NA19679","NA19682","NA19717","NA19720","NA19723","NA19726","NA19729","NA19732","NA19735","NA19741","NA19747","NA19750","NA19756","NA19759","NA19762","NA19771","NA19774","NA19777","NA19780","NA19783","NA19786","NA19789","NA19792","NA19795"