#############################################
# Logistic regression + Odds Ratios (R)
# Enriched Listeria soil + weather dataset
#############################################

# Packages ---------------------------------------------------------------
library(tidyverse)     # data wrangling + readr
library(janitor)       # clean_names()
library(car)           # Anova(Type II LR tests)
library(broom)         # tidy model outputs
library(performance)   # check_collinearity()

# 1) Read and inspect data ----------------------------------------------
df_rich <- read_csv(file.choose()) %>%
  clean_names()

# Quick check of column names (useful for debugging)
names(df_rich)


# 2) Create binary outcome (presence/absence) ---------------------------
# label = 1 if at least one Listeria isolate was obtained, else 0
df_rich <- df_rich %>%
  mutate(label = as.integer(number_of_listeria_isolates_obtained > 0))

# Sanity check: class balance
df_rich %>% count(label)


# 3) Define columns to drop (avoid leakage + IDs + metadata) ------------
# These variables should NOT be used as predictors because:
# - they are the outcome itself or derived from the outcome (leakage)
# - they are identifiers / administrative fields
# - they are not intended as explanatory variables
drop_cols <- c(
  "label",
  "x1",  # often an index-like artifact from CSV export
  "number_of_listeria_isolates_obtained",
  "number_of_listeria_isolates_selected_for_wgs_i_e_number_of_isolates_with_unique_sig_b",
  "if_selected_for_soil_property_analysis_yes_y_no_n",
  "index",
  "sample_id",
  "sampling_grid",
  "date_ymd",
  "wx_key",     # may not exist; safe to include anyway
  "us_state"    # treat geography via lat/long/elevation instead of state label
)


# 4) Build modeling dataset (numeric predictors only) --------------------
# Why numeric only?
# - logistic regression can handle factors, but interpreting ORs is cleaner
#   when predictors are numeric and/or explicitly coded.
x_df <- df_rich %>%
  select(-any_of(drop_cols)) %>%  # drop leakage/IDs safely
  select(where(is.numeric))       # keep numeric predictors only

# Final dataset: label + predictors
df_model <- bind_cols(
  df_rich %>% select(label),
  x_df
)

glimpse(df_model)


# 5) Fit FULL logistic regression model ---------------------------------
# Model form: logit(P(label=1)) = beta0 + beta1*x1 + beta2*x2 + ...
mod_glm_full <- glm(
  label ~ .,
  data = df_model,
  family = binomial(link = "logit")
)

summary(mod_glm_full)

# Type II LR tests: “Does each variable matter after adjusting for others?”
# (Good for multivariable inference)
Anova(mod_glm_full, type = 2, test = "LR")


# 6) Odds ratios (OR) for a meaningful “c-unit” increase -----------------
# OR for a 1-unit increase: exp(beta)
# OR for a c-unit increase: exp(c*beta) = (exp(beta))^c
c_increase <- 10  # e.g., +10 units (useful for percent-type variables)

or_c_table <- broom::tidy(mod_glm_full, conf.int = TRUE) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    OR_1 = exp(estimate),
    OR_c = exp(c_increase * estimate),
    CI_low_c  = exp(c_increase * conf.low),
    CI_high_c = exp(c_increase * conf.high),
    CI_c = sprintf("[%.3f, %.3f]", CI_low_c, CI_high_c)
  ) %>%
  select(term, OR_1, OR_c, CI_c, p.value) %>%
  arrange(p.value)

print(or_c_table, n = 28)


# 7) Fit a REDUCED logistic regression model ----------------------------
# Goal of reduced model:
# - improve interpretability
# - reduce collinearity and unstable ORs
# - focus on “core drivers” supported by SHAP + LR tests
#
# NOTE: Here we fit directly on df_rich so we can specify columns by name.
# If you want to guarantee the exact same preprocessing as df_model,
# you can instead build a reduced df_model and fit on that.
mod_glm_reduced <- glm(
  label ~
    longitude +
    elevation_m +
    temp_mean +
    rh_max +
    sampling_date,
  data = df_rich,
  family = binomial(link = "logit")
)

summary(mod_glm_reduced)

# Type II LR tests for reduced model
Anova(mod_glm_reduced, type = 2, test = "LR")


# 8) Compare FULL vs REDUCED model --------------------------------------
# Lower AIC = better trade-off between fit and complexity
AIC(mod_glm_full, mod_glm_reduced)

# Likelihood Ratio Test: tests whether the reduced model loses information
# H0: reduced model fits as well as full model
anova(mod_glm_full, mod_glm_reduced, test = "Chisq")


# 9) Collinearity diagnostics -------------------------------------------
# Helps detect redundant predictors (high VIF)
check_collinearity(mod_glm_reduced)

# If temp_mean shows high VIF with another temperature metric,
# consider keeping ONLY one temperature variable (e.g., temp_max OR temp_mean).
