"""Extract donor dosages for the BIN1 cis-eQTL variant
2_127135234_C_T (top AD-locus eQTL in the paper's untreated
nominal table, p=7.05e-22, AF=0.392) from the record VCF by
streaming scan; writes data/perez_bin1_genotypes.csv."""
import csv, gzip

TARGET_CHR, TARGET_POS = "2", 127135234
samples, row = None, None
with gzip.open("data/microglia_samples.vcf.gz", "rt") as fh:
    for line in fh:
        if line.startswith("#CHROM"):
            samples = line.rstrip("\n").split("\t")[9:]
            continue
        if samples is None or line.startswith("#"):
            continue
        if line.startswith("2\t"):
            parts = line.rstrip("\n").split("\t", 3)
            if int(parts[1]) == TARGET_POS:
                row = line.rstrip("\n").split("\t")
                break
            if int(parts[1]) > TARGET_POS:
                break
assert samples and row, "variant not found"
fmt = row[8].split(":")
gt_i = fmt.index("GT")
out = []
for name, field in zip(samples, row[9:]):
    gt = field.split(":")[gt_i]
    dos = gt.replace("|", "/").split("/")
    out.append((name, sum(int(a) for a in dos if a != ".")))
with open("data/perez_bin1_genotypes.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["donor_id", "bin1_dosage"])
    w.writerows(out)
from collections import Counter
print("samples:", len(out), "dosage counts:", Counter(d for _, d in out))
