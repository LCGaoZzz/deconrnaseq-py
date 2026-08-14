# R baseline export + timing for DeconRNASeq 1.50.0 (GPL-2).
# Usage: Rscript run_r_baseline_timed.R <data_dir> <output_dir> [reps]
# Exports:
#   mixtures_exact_r_unscaled.csv / mixtures_exact_r_scaled.csv
#   mixtures_noisy_r_unscaled.csv / mixtures_noisy_r_scaled.csv
#   r_timing.csv  (median/p95 seconds over `reps` repeats after 1 warmup)
# Timing EXCLUDES: library load, CSV reading. Includes three scopes:
#   full_function : DeconRNASeq(..., fig=FALSE) including PCA diagnostics
#   core_df       : scaling + per-sample lsei loop, inputs as data.frame (as in package)
#   core_matrix   : same loop with matrix inputs (R best case)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript run_r_baseline_timed.R <data_dir> <output_dir> [reps]")
data.dir <- args[[1]]
output.dir <- args[[2]]
reps <- if (length(args) >= 3) as.integer(args[[3]]) else 11L
dir.create(output.dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages(library(DeconRNASeq))

read.matrix <- function(file) read.csv(file.path(data.dir, file), row.names = 1, check.names = FALSE)

signatures <- read.matrix("reference_signatures.csv")
cases <- c("mixtures_exact.csv", "mixtures_noisy.csv")

# ---- core loop replicated verbatim from DeconRNASeq 1.50.0 (R/DeconRNASeq.R) ----
core.solve <- function(x.data, x.signature, use.scale) {
  common.signature <- rownames(x.signature) %in% rownames(x.data)
  common.data <- rownames(x.data) %in% rownames(x.signature)
  x.data <- x.data[common.data, ]
  x.signature <- x.signature[common.signature, ]
  x.subdata <- x.data[rownames(x.signature), ]
  Numofx <- ncol(x.signature)
  AA <- if (use.scale) scale(x.signature) else x.signature
  EE <- rep(1, Numofx); FF <- 1
  GG <- diag(nrow = Numofx); HH <- rep(0, Numofx)
  out.all <- c()
  for (i in colnames(x.subdata)) {
    BB <- x.subdata[, i]
    if (use.scale) BB <- scale(BB)
    out <- lsei(AA, BB, EE, FF, GG, HH)
    out.all <- rbind(out.all, out$X)
  }
  out.all
}

timeit <- function(fun, reps) {
  invisible(fun())  # warmup
  times <- numeric(reps)
  for (r in seq_len(reps)) {
    t0 <- proc.time()[["elapsed"]]
    invisible(fun())
    times[r] <- proc.time()[["elapsed"]] - t0
  }
  c(median = median(times), p95 = as.numeric(quantile(times, 0.95, names = FALSE)), n = reps)
}

timing.rows <- list()

for (case in cases) {
  mixtures <- read.matrix(case)
  stem <- sub("\\.csv$", "", case)
  mixtures.mat <- as.matrix(mixtures)
  signatures.mat <- as.matrix(signatures)

  for (use.scale in c(FALSE, TRUE)) {
    suffix <- if (use.scale) "scaled" else "unscaled"

    # ---- export reference results (official package, full function) ----
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
    write.csv(estimates, file.path(output.dir, paste0(stem, "_r_", suffix, ".csv")))

    # ---- timing, three scopes ----
    timing.rows[[paste(stem, suffix, "full_function", sep = ".")]] <-
      data.frame(case = stem, use_scale = use.scale, scope = "full_function",
                 t(timeit(function() DeconRNASeq(datasets = mixtures, signatures = signatures,
                                                 checksig = FALSE, known.prop = FALSE,
                                                 use.scale = use.scale, fig = FALSE), reps)))
    timing.rows[[paste(stem, suffix, "core_df", sep = ".")]] <-
      data.frame(case = stem, use_scale = use.scale, scope = "core_df",
                 t(timeit(function() core.solve(mixtures, signatures, use.scale), reps)))
    timing.rows[[paste(stem, suffix, "core_matrix", sep = ".")]] <-
      data.frame(case = stem, use_scale = use.scale, scope = "core_matrix",
                 t(timeit(function() core.solve(mixtures.mat, signatures.mat, use.scale), reps)))
  }
}

timing <- do.call(rbind, timing.rows)
rownames(timing) <- NULL
write.csv(timing, file.path(output.dir, "r_timing.csv"), row.names = FALSE)

cat("sessionInfo:\n")
print(sessionInfo())
cat("BLAS/LAPACK:\n")
print(extSoftVersion())
cat("DONE\n")
