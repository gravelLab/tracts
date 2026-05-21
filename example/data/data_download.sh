#!/usr/bin/env bash

# Download ancestry deconvolution archives
# for the 6 admixed Phase 3 1000 Genomes populations.
#
# Source:
# https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20140818_ancestry_deconvolution/
# This file was created using AI assistance.  
set -euo pipefail

BASE_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20140818_ancestry_deconvolution"

POPS=(
    "ACB"
    "ASW"
    "CLM"
    "MXL"
    "PEL"
    "PUR"
)

OUTDIR="1000G_ancestry_deconvolution"

mkdir -p "${OUTDIR}"

download_file() {
    local url="$1"
    local outfile="$2"

    echo
    echo "Downloading:"
    echo "  ${url}"

    if [[ -s "${outfile}" ]]; then
        echo
        echo "File already exists, skipping:"
        echo "  ${outfile}"
        return 0
    fi


    if command -v wget >/dev/null 2>&1; then
        wget -c -O "${outfile}" "${url}"

    elif command -v curl >/dev/null 2>&1; then
        curl -L --retry 3 -o "${outfile}" "${url}"

    else
        echo "ERROR: wget or curl is required."
        exit 1
    fi
}

for POP in "${POPS[@]}"; do

    echo
    echo "======================================="
    echo "Population: ${POP}"
    echo "======================================="

    POP_DIR="${OUTDIR}"

    ZIP_NAME="${POP}_phase3_ancestry_deconvolution.zip"

    URL="${BASE_URL}/${ZIP_NAME}"

    ZIP_PATH="${POP_DIR}/${ZIP_NAME}"

    download_file "${URL}" "${ZIP_PATH}"

    echo
    echo "Extracting ${ZIP_NAME} ..."

    unzip -o "${ZIP_PATH}" -d "${POP_DIR}"
    
    BEDDIR="${OUTDIR}/${POP}/PopPhased/bed_files"
    echo  "${BEDDIR}"
    if compgen -G "${BEDDIR}/*.bed" > /dev/null; then
        echo "BED files already extracted for ${POP}, skipping."
    else
        tar -xzvf "${BEDDIR}/${POP}_bed.tar.gz" -C "${BEDDIR}"
    fi
    
    #reorganize directories for consistency across populations
    if [[ -d "${BEDDIR}/${POP}/PopPhased/" ]]; then
        
        echo "mv ${BEDDIR}/${POP}/PopPhased/* ${BEDDIR}/"
        mv "${BEDDIR}/${POP}/PopPhased/"* "${BEDDIR}/"
        rmdir "${BEDDIR}/${POP}/PopPhased"
    else
        echo "File not found: "${BEDDIR}/${POP}/PopPhased/""
    fi

    

    echo
    echo "Finished ${POP}"
done

echo
echo "======================================="
echo "All downloads complete."
echo "======================================="

echo
echo "Example files:"
find "${OUTDIR}" -type f | head -20