#!/bin/bash
#for i in  
#do 
#awk '{gsub(/BER/,"AFR"); print}' $i > "AFAM"$i 
#done

for ind in "NA19700" "NA19701"  "NA19704"   "NA19703"  "NA19819"   "NA19818" "NA19835"   "NA19834" "NA19901"   "NA19900"  "NA19909"   "NA19908" "NA19917"   "NA19916"  "NA19713"   "NA19982" "NA20127"   "NA20126" "NA20357"   "NA20356"
do 
echo $ind
mv "AFAM"$ind"_A.bed" $ind"_A.bed"
mv "AFAM"$ind"_B.bed" $ind"_B.bed"
done
