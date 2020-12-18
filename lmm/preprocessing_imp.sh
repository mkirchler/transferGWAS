#!/bin/bash

# TODO fill in the path to your plink2 install, if not in global path
# alias plink2=/path/to/plink2

# TODO fill in space-separated file with columns FID and IID, no header
INDIV=/path/to/indiv.txt

# Hardy-Weinberg p-value (1e-12)
HWE_PVAL=0.000000000001

# minor allele frequency
MAF=0.001

# number of threads to use
THREADS=40



for i in {1..22}; do
	# INPUT=${IMP_FN}${i}${IMP_SUFF}
	INPUT="/path/to/imputed/chr${i}.bgen"
	SAMPLE="/path/to/imputed/chr${i}.sample"
	OUTPUT="/path/to/output/chr${i}"

	plink2 --bgen ${INPUT} 'ref-first' --sample ${SAMPLE} --keep ${INDIV} --maf ${MAF} --hwe ${HWE_PVAL} --threads ${THREADS} --export bgen-1.2 bits=8 --out ${OUTPUT}

done
