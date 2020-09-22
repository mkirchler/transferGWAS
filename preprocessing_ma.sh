# TODO fill in the path to your plink2 install, if not in global path
# alias plink2=/path/to/plink2

# TODO fill in space-separate file with columns FID and IID, no header
INDIV=/path/to/indiv.txt



# very soft filtering parameters:
# minor allele frequency
MAF=0.001

# Hardy-Weinberg p-value
HWE_PVAL=0.001

# LD-pruning, R^2, window-size & number of steps (see plink2 documentation for details)
LD_THRESHOLD=0.8
LD_WINDOW="500 kb"
LD_STEP=1



TMP_DIR=tmp_dir_del/
mkdir ${TMP_DIR}

for i in {1..22}; do
    # TODO fill in path to i-th bed/bim/fam files
    INPUT="/path/to/microarray/chr${i}"
    # TODO fill in where to save i-th bed/bim/fam files
	OUTPUT="/path/to/output/chr${i}"



    TMPFILE=${TMP_DIR}/chr${i}.tmp
	LD_TMP=${TMP_DIR}/ld
	# individuals, Hardy-Weinberg, and MAF 
	plink2 --bfile ${INPUT} --keep ${INDIV} --hwe ${HWE_PVAL} --maf ${MAF} --make-bed --out ${TMPFILE}
	# LD pruning
	plink2 --bfile ${TMPFILE} --indep-pairwise ${LD_WINDOW} ${LD_STEP} ${LD_THRESHOLD} --out ${LD_TMP}
	plink2 --bfile ${TMPFILE} --extract ${LD_TMP}.prune.in --make-bed --out ${OUTPUT}
done

rm -r ${TMP_DIR}
