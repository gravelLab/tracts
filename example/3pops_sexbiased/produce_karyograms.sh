#!/usr/bin/env bash
# Documents how to regenerate an individual's karyogram figure (plot_karyogram.py), colored
# with the same ancestry colors as this directory's other plots (produce_plots_and_tables.sh).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python plot_karyogram.py \
    ../data/1000G_ancestry_deconvolution/ASW/PopPhased/bed_files/NA19700_A_final.bed \
    ../data/1000G_ancestry_deconvolution/ASW/PopPhased/bed_files/NA19700_B_final.bed \
    --output ./plots/NA19700_final
