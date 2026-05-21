#!/usr/bin/env bash




POPS=(
    "ACB"
    "ASW"
    "CLM"
    "MXL"
    "PEL"
    "PUR"
)

echo "======================================"
echo "Checking individual mismatches"
echo "======================================"

for POP in "${POPS[@]}"; do


    BED_DIR="./1000G_ancestry_deconvolution/${POP}/PopPhased/bed_files"
    INDIV_FILE="${BED_DIR}/individuals.txt"
    UNRELATED_FILE="${BED_DIR}/individuals_unrelated.txt"
    PANEL="metadata/integrated_call_samples_v3.20130502.ALL.panel"
    UNRELATED_MALES="${BED_DIR}/males_unrelated.txt"
    
    
    ls "${BED_DIR}" | grep _A_ |cut -f 1 -d "_" > "${INDIV_FILE}"

    

    echo
    echo "Population: ${POP}"

    if [[ ! -f "${INDIV_FILE}" ]]; then
        echo "  Missing file: ${INDIV_FILE}"
        continue
    fi

    MISMATCHES=$(
        awk '
        NR==FNR {
            seen[$1]=1
            next
        }

        !($1 in seen) {
            print $1
        }
        ' "${PANEL}" "${INDIV_FILE}"
    )

    if [[ -z "${MISMATCHES}" ]]; then
        echo "  No mismatches found."
    else
        echo " Mismatches:"
        echo "${MISMATCHES}" | sed 's/^/    /'
    fi

#relateds:"
    
    RELATEDS=$(awk '
        NR==FNR { seen[$1]=1; next 
        } ($1 in seen) { 
        print $1 
        } ' metadata/20140625_related_individuals.txt "${INDIV_FILE}"
        )
    echo " Relateds:"
    echo "${RELATEDS}" | sed 's/^/    /'

    awk '
        NR==FNR  { 
        seen[$1]=1; next 
        } !($1 in seen) {
            if (out == "") 
                {
                out = $1
                } 
            else 
                {
                out = out "," $1
                }
        }

END {
    print out
} ' metadata/20140625_related_individuals.txt "${INDIV_FILE}" > "${UNRELATED_FILE}"


awk '
NR==FNR {
    sex[$1] = $4
    next
}
FNR==1 {
    n = split($0, ids, ",")
    for (i = 1; i <= n; i++) {
        id = ids[i]
        if (id in sex && sex[id] == "male") {
            out = (out == "" ? id : out "," id)
        }
    }
}

END { print out }
' "${PANEL}" "${UNRELATED_FILE}" > "${UNRELATED_MALES}"

done