#!/usr/bin/env bash

set -euo pipefail

BASE_URL="https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"

OUTDIR="metadata"

FILES=(
    "integrated_call_samples_v3.20130502.ALL.panel"
    "20140625_related_individuals.txt"
)

mkdir -p "${OUTDIR}"

download_file () {
    local url="$1"
    local outfile="$2"

    echo "Downloading:"
    echo "  ${url}"

    if command -v wget >/dev/null 2>&1; then
        wget -c -O "${outfile}" "${url}"

    elif command -v curl >/dev/null 2>&1; then
        curl -L "${url}" -o "${outfile}"

    else
        echo "ERROR: wget or curl is required."
        exit 1
    fi
}

echo
echo "======================================"
echo "Downloading 1000 Genomes metadata"
echo "======================================"

for FILE in "${FILES[@]}"; do

    URL="${BASE_URL}/${FILE}"

    OUTFILE="${OUTDIR}/${FILE}"

    download_file "${URL}" "${OUTFILE}"

done

echo
echo "Done."
echo
echo "Files saved to:"
echo "  ${OUTDIR}/"

ls -lh "${OUTDIR}"