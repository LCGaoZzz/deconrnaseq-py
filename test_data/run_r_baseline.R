args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Usage: Rscript run_r_baseline.R <output_dir>")
}

suppressPackageStartupMessages(library(DeconRNASeq))

all.args <- commandArgs(trailingOnly = FALSE)
file.arg <- grep("^--file=", all.args, value = TRUE)
if (length(file.arg) != 1) {
  stop("Cannot determine the script path; run this file with Rscript")
}
script.path <- normalizePath(sub("^--file=", "", file.arg[[1]]), mustWork = TRUE)
root <- dirname(script.path)
output.dir <- args[[1]]
dir.create(output.dir, recursive = TRUE, showWarnings = FALSE)

read.matrix <- function(file) {
  read.csv(file.path(root, file), row.names = 1, check.names = FALSE)
}

signatures <- read.matrix("reference_signatures.csv")
cases <- c("mixtures_exact.csv", "mixtures_noisy.csv")

for (case in cases) {
  mixtures <- read.matrix(case)
  for (use.scale in c(FALSE, TRUE)) {
    result <- DeconRNASeq(
      datasets = mixtures,
      signatures = signatures,
      checksig = FALSE,
      known.prop = FALSE,
      use.scale = use.scale,
      fig = FALSE
    )
    estimates <- result$out.all
    rownames(estimates) <- colnames(mixtures)
    colnames(estimates) <- colnames(signatures)
    stem <- sub("\\.csv$", "", case)
    suffix <- if (use.scale) "scaled" else "unscaled"
    write.csv(estimates, file.path(output.dir, paste0(stem, "_r_", suffix, ".csv")))
  }
}
