## Local Ancestry Inference for 1000 Genomes Project (Phase 3)

### Overview

This repository contains **haploid local ancestry inference (LAI) tracts** for Phase 3 admixed populations from the **1000 Genomes Project**. The data were generated in **2014** by the 1000 Genomes Project Admixture Working Group and downloaded from the [official 1000 Genomes FTP server](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20140818_ancestry_deconvolution/
) on **December 5, 2025**. All tracts were generated with respect to **hg19**.

### Populations Included

The following admixed populations are included:

- **ACB** – Afro-Caribbean from Barbados
- **ASW** – African American from Southwest US
- **CLM** – Colombians from Medellín
- **MXL** – Mexicans from Los Angeles
- **PEL** – Peruvians from Lima
- **PUR** – Puerto Ricans

### File Organization

For each population, data are distributed in compressed archives:

```
*_phase3_ancestry_deconvolution.zip
```

Each archive contains **two haploid BED files per individual** (one per chromosome copy), for example:

```
PUR/PopPhased/bed_files/HG00553_A.bed
PUR/PopPhased/bed_files/HG00553_B.bed
```

### Directory Structure

Each population directory (`[POP]`) follows the structure below:

```
[POP]/
├── PopPhased/
│   ├── alleles_rephased/
│   ├── bed_files/
│   ├── karyograms/
│   └── lai_global_*.txt
├── TrioPhased/
│   ├── bed_files/
│   ├── karyograms/
│   └── lai_global_*.txt
└── rfmix_input/
```

Where:

- **PopPhased**: Phase-corrected LAI calls,
- **TrioPhased**: Phase-uncorrected LAI calls,
- **bed_files**: Collapsed haploid ancestry tracts,
- **karyograms**: PNG visualizations of haploid ancestry along chromosomes,
- **rfmix_input**: Input files used to generate LAI calls (alleles, maps, samples, classes).

---

### BED File Format

Each haploid BED file is **tab-delimited** with the following six columns:

| Column | Description |
|:------|:------------|
| 1 | Chromosome number (1–22) |
| 2 | Start physical position (0-based, bp) |
| 3 | End physical position (1-based, bp) |
| 4 | Haploid ancestry call |
| 5 | Start genetic position (cM) |
| 6 | End genetic position (cM) |

### Ancestry Codes

| Code | Meaning |
|:-----|:--------|
| AFR | African |
| EUR | European |
| NAT | Native American |
| UNK | Uncertain / low-confidence (posterior < 0.9) |
| centromere | Centromeric regions (3 Mb each) |
| miscall | Masked region with high miscall rate |

### Masked Region on Chromosome 15

The following region is masked due to high empirical miscall rates:

| Chromosome | Start (bp) | End (bp) | Code | cM start | cM end |
|:-----------|:-----------|:---------|:------|:---------|:--------|
| 15 | 20,071,673 | 22,422,348 | miscall | 0.00598 | 23.70848 |

A BED file containing **all masked regions** (centromeres + miscall) is provided as:

```
cent_miscall.bed
```

---

### Global Ancestry Proportions

For each population, global ancestry proportions are provided in files named:

```
lai_global_*.txt
```

### Columns

| Column | Description |
|:------|:------------|
| 1 | 1000 Genomes individual ID |
| 2 | Proportion African ancestry |
| 3 | Proportion European ancestry |
| 4 | Proportion Native American ancestry |
| 5 | Proportion Unknown (UNK only) |

Example definition for European ancestry:

```
(sum[EUR/EUR] + (sum[EUR/AFR] + sum[EUR/NAT]) / 2)
--------------------------------------------------
           sum[EUR + AFR + NAT + UNK]
```

Note: `UNK` excludes regions masked in all individuals (centromere and miscall).

---

### Methodology

#### Reference Panels and Phasing

- Phased haplotypes were obtained from the 1000 Genomes Phase 3 FTP site (SHAPEIT2).
- Native American reference samples (>99% NAT ancestry) were phased similarly.
- Reference panel included:
  - 50 CEU individuals
  - 50 YRI individuals
  - 43 Native American individuals

---

#### Local Ancestry Inference

Local ancestry was inferred using **RFMix v1.5.4**, a discriminative LAI method based on allele frequency differences among ancestral populations.

Two inference modes were used:

**TrioPhased**

- No phase correction
- RFMix options:
  - `-w 0.5`
  - `--forward-backward`

**PopPhased**

- Includes phase correction via EM
- RFMix options:
  - `-w 0.5`
  - `--forward-backward`
  - `--use-reference-panels-in-EM`
  - `-e 5`

Calls from the **5th EM iteration** were retained.

#### Post-processing

- Haploid ancestry calls were collapsed into tracts
- Karyograms were generated per individual
- Global ancestry proportions were computed

Parsing scripts used in this pipeline are available [here](https://github.com/armartin/ancestry_pipeline).

---

### References

1. Mao et al. *A genomewide admixture mapping panel for Hispanic/Latino populations*. AJHG 80, 1171 (2007).
2. O’Connell et al. *A General Approach for Haplotype Phasing across the Full Spectrum of Relatedness*. PLoS Genet. 10, e1004234 (2014).
3. Maples et al. *RFMix: A Discriminative Modeling Approach for Rapid and Robust Local-Ancestry Inference*. AJHG 93, 278–288 (2013).
4. The Native American training data (Affymetrix 6.0 genotype data) is also available from [this ftp site](ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130711_native_american_admix_train/).

### Contact

Data generated by the **1000 Genomes Project Admixture Working Group** (2014):

  * Alicia Martin, Brian Maples, and Carlos Bustamante at Stanford University.
  * Simon Gravel and Soheil Baharian at McGill University.
  * Eimear Kenny at Icahn School of Medicine at Mount Sinai.

For questions related to the original dataset contact Alicia Martin — armartin@stanford.edu.

