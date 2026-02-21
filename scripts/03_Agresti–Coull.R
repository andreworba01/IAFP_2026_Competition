library(fitdistrplus)
library(extraDistr)
library(binom)
library(dplyr)

#############################################
# Goal of this script
# -------------------------------------------
# Using soil presence/absence data for Listeria:
#  1) Compute state-level sample size (n), positives (w), and prevalence (w/n)
#  2) Quantify uncertainty in prevalence using Agresti–Coull (AC) 95% CI
#  3) Convert prevalence uncertainty into *concentration* estimates (CFU/g)
#     using a Beta–Binomial + Poisson framework, via Monte Carlo simulation
#############################################

# ------------------------------------------------------------
# 0) Read and inspect data
# ------------------------------------------------------------
library(readr)
library(janitor)
library(dplyr)
library(binom)
library(sf)

# Read a CSV chosen interactively, then standardize column names
df <- read_csv("C:/Users/andre/Downloads/ListeriaSoil_clean_binary (1).csv") %>%
  clean_names()

# Quick look at structure (columns, types, first values)
glimpse(df)

# ------------------------------------------------------------
# 1) Global settings for reproducibility and simulation
# ------------------------------------------------------------
set.seed(473)     # Ensures simulation results are reproducible

V <- 25           # Sample mass in grams (used in Poisson conversion to CFU/g)
alpha <- 0.05     # Significance level => 95% confidence intervals
n_sims <- 100000  # Number of Monte Carlo draws per state (higher = smoother/stabler)

# ------------------------------------------------------------
# 2) Helper function: Beta–Binomial prevalence -> Poisson concentration
# ------------------------------------------------------------
simulate_BR_poisson <- function(pU, n, V = 1, n_sims = 10000) {
  # PURPOSE:
  #   Given an estimated prevalence (pU) and sample size (n),
  #   simulate plausible prevalence values from a Beta posterior and convert them
  #   into concentrations using a Poisson assumption.
  #
  # INPUTS:
  #   pU     = prevalence estimate (here we typically use an upper CI bound)
  #   n      = number of samples for that state
  #   V      = grams of soil per sample (e.g., 25 g)
  #   n_sims = number of simulations (Monte Carlo draws)
  #
  # OUTPUT:
  #   A list with:
  #     - beta_a, beta_b: Beta posterior parameters
  #     - Cs_Li_mean: mean simulated concentration (CFU/g)
  #     - sim_q95: 95th percentile concentration (upper-tail metric)
  
  eps <- 1e-6
  
  # Force prevalence into (0,1) to avoid issues with log(0) later
  pU <- pmin(pmax(pU, eps), 1 - eps)
  
  # Convert prevalence to an approximate count of positives.
  # NOTE: This approximates E(w) ≈ pU*n. 
  # and skip rounding.
  x <- round(pU * n)
  
  # Beta posterior under a uniform prior Beta(1,1):
  #   p | data ~ Beta(x + 1, (n - x) + 1)
  a <- x + 1
  b <- (n - x) + 1
  
  # Draw prevalence samples from the posterior
  P_sim <- rbeta(n_sims, shape1 = a, shape2 = b)
  P_sim <- pmin(P_sim, 1 - eps)  # keep away from 1 to avoid log(0)
  
  # Convert prevalence (probability of at least one CFU in V grams)
  # into a mean concentration using the Poisson "presence" relationship:
  #
  #   P(positive) = 1 - exp(-lambda * V)
  #   => lambda (CFU/g) = -log(1 - P) / V
  Cs_Li <- -log(1 - P_sim) / V
  
  # Upper-tail metric often used for exposure (e.g., 95th percentile)
  q95 <- as.numeric(quantile(Cs_Li, 0.95))
  
  # Return summary statistics
  list(
    beta_a = a,
    beta_b = b,
    Cs_Li_mean = mean(Cs_Li),
    sim_q95 = q95
  )
}

# ------------------------------------------------------------
# 3) STATE-LEVEL COUNTS (n, w, prevalence)
# ------------------------------------------------------------
# Assumptions about your spatial data:
#   - pts_with_state is an sf object (points) already spatially joined to state polygons
#   - NAME is the state name column
#   - label == 1 indicates Listeria presence; label == 0 indicates absence
#
# This block produces a tidy summary per state:
#   n = number of samples in that state
#   w = number of positives
#   prevalence = w/n
state_counts <- pts_with_state %>%
  st_drop_geometry() %>%          # remove geometry for faster dplyr operations
  filter(!is.na(NAME)) %>%        # keep only rows with state name
  group_by(NAME) %>%
  summarize(
    n = n(),                      # total samples in state
    w = sum(label == 1, na.rm = TRUE),  # number of positives
    prevalence = w / n,           # observed prevalence
    .groups = "drop"
  )

# ------------------------------------------------------------
# 4) PREVALENCE UNCERTAINTY: Agresti–Coull (AC) 95% CI
# ------------------------------------------------------------
# We compute the AC confidence interval per state. Then:
#   - ci_low_raw = lower bound
#   - pU_raw     = upper bound (we use this as pU for conservative modeling)
#
# Why clip/cap?
#   - Some methods may produce values slightly outside [0,1] due to rounding/numerics.
#   - We also create pU_cap in (0,1) for log() stability in the Poisson conversion.
state_ci <- state_counts %>%
  rowwise() %>%
  mutate(
    # binom.confint returns a data.frame; we store it as a list-column
    ci = list(binom.confint(
      x = w,
      n = n,
      conf.level = 1 - alpha,   # 95% CI if alpha=0.05
      methods = "ac"            # Agresti–Coull
    )),
    ci_low_raw = ci$lower,
    pU_raw     = ci$upper
  ) %>%
  ungroup() %>%
  mutate(
    eps = 1e-6,
    
    # Ensure CI bounds are within probability range
    ci_low = pmax(ci_low_raw, 0),
    pU     = pmin(pU_raw, 1),
    
    # "Safe" version of upper bound strictly inside (0,1) for log(1 - p) operations
    pU_cap = pmin(pmax(pU_raw, eps), 1 - eps)
  ) %>%
  select(-eps)

# ------------------------------------------------------------
# 5) SIMULATE concentration estimates per state
# ------------------------------------------------------------
# For each state:
#   - Use pU_cap (conservative, stable upper-bound prevalence)
#   - Use state-specific n and global V and n_sims
# Output:
#   - Cs_Li_mean: mean simulated concentration (CFU/g)
#   - sim_q95: 95th percentile concentration (CFU/g)
#   - beta_a, beta_b: posterior parameters (useful for transparency)
results <- state_ci %>%
  rowwise() %>%
  mutate(
    sim = list(simulate_BR_poisson(
      pU = pU_cap,
      n  = n,
      V  = V,
      n_sims = n_sims
    )),
    Cs_Li_mean = sim$Cs_Li_mean,
    sim_q95    = sim$sim_q95,
    beta_a     = sim$beta_a,
    beta_b     = sim$beta_b
  ) %>%
  ungroup() %>%
  mutate(
    # Create a readable CI string for display/reporting
    ci_95 = sprintf("[%.3f, %.3f]", ci_low, pU)
  ) %>%
  select(
    state = NAME,
    n,
    prevalence,
    ci_low,
    pU,
    pU_cap,
    ci_95,
    beta_a,
    beta_b,
    Cs_Li_mean,
    sim_q95
  ) %>%
  arrange(desc(prevalence))

# View the full table and the last few rows
results
tail(results)

# ------------------------------------------------------------
# 6) Quick diagnostics / sanity checks
# ------------------------------------------------------------
# Distribution of capped pU used in simulation
summary(results$pU_cap)

# How many unique pU_cap values exist (helpful to see if many states share similar values)
length(unique(results$pU_cap))


############################################################
# 7) Purpose: Empirical ("true") CI coverage by state
# ----------------------------------------------------------
# Even though we *label* a confidence interval as "95% CI",
# the actual probability that it contains the true prevalence
# can deviate from 0.95—especially for small n or extreme p.
#
# binom.coverage() estimates the *true coverage* (long-run
# proportion of times the CI would contain the true p) for a
# given method, sample size n, and assumed true prevalence p.
#
# Here we compute this per state using:
#   - p = pU (state-specific prevalence value used in our table)
#   - n = state sample size
#   - method = Agresti–Coull
############################################################


results_conf <- results %>%
  rowwise() %>%  # Evaluate coverage row-by-row (each row = one state)
  mutate(
    # true_coverage = the *actual* confidence level achieved by the CI method
    # for the given state sample size (n) and assumed true prevalence (p).
    #
    # Interpretation:
    #   true_coverage < 0.95  => under-coverage (CI too narrow; overconfident)
    #   true_coverage ~ 0.95  => near-nominal coverage (desired)
    #   true_coverage > 0.95  => over-coverage (CI conservative; wider than needed)
    true_coverage = binom.coverage(
      p = pU,                 # assumed "true" prevalence for this evaluation
      n = n,                  # state-specific number of samples
      conf.level = 0.95,      # nominal confidence level
      method = "agresti-coull" # CI method being evaluated
    )$coverage                # extract the numeric coverage value
  ) %>%
  ungroup()  # return to a normal (non-rowwise) tibble

# Summarize the resulting columns, including true_coverage distribution
summary(results_conf)

