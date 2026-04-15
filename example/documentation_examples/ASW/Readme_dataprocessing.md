# Data Processing Notes

Created a list of all individuals in the local ancestry calls downloaded from **1000G**. Downloaded 1000G sex variables from

```
integrated_call_samples_v3.20130502.ALL.panel
```

here:

```
https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
```

Checked which individuals were not included in the panel.

---

## Individuals not included in the panel

```bash
awk '
NR==FNR { seen[$1]=1; next }
!($1 in seen) { print $1 }
' integrated_call_samples_v3.20130502.ALL.panel ./ASW/TrioPhased/individuals.txt
```

```
NA19985
NA20322
NA20336
NA20341
NA20344
```

---

These are all relateds, according to `20140625_related_individuals.txt` from:

```
https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/
```

```bash
awk '
NR==FNR { seen[$1]=1; next }
($1 in seen) { print $1 }
' 20140625_related_individuals.txt ./ASW/TrioPhased/individuals.txt
```

```
NA19985
NA20322
NA20336
NA20341
NA20344
```

---

So these individuals are included in local ancestry, but will not be independent. We need to remove them from the call list.

```bash
awk '
NR==FNR { seen[$1]=1; next }
!($1 in seen) { print $1 }
' 20140625_related_individuals.txt ./ASW/TrioPhased/individuals.txt
```

```
NA19985
NA20322
NA20336
NA20341
NA20344
```

```bash
awk '
NR==FNR { seen[$1]=1; next }
!($1 in seen) { print $1 }
' 20140625_related_individuals.txt ./ASW/TrioPhased/individuals.txt \
> ./ASW/TrioPhased/individuals_unrelated.txt
```

---

## Output formatting

Output as CSV to copy in driver file:

```bash
awk '
NR==FNR { seen[$1]=1; next }
($1 in seen) { printf "\"%s\"," , $1 }
' 20140625_related_individuals.txt ./ASW/TrioPhased/individuals.txt
```

Output males in similar fashion:

```bash
awk '
NR==FNR { seen[$1]=1; next }
($1 in seen) { printf "\"%s\"," , $1 }
' ./males_ASW.panel ./ASW/TrioPhased/individuals_unrelated.txt
```

