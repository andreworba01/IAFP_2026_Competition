# Listeria in Soil — IAFP AI Benchmarking Student Competition 🏆🦠

🥬 This repository contains an end-to-end, reproducible **machine learning pipeline** developed for the **IAFP AI Benchmarking Student Competition on Predictive Food Safety Models**. The project addresses the challenge theme:

> **Predicting pathogen presence in food production environments**

Using a nationwide GIS-based dataset of U.S. soil samples, this work focuses on predicting the **presence or absence of *Listeria spp.* in soil** based on environmental, soil, climate, land-use, and geographic variables. The goal is to demonstrate how **AI/ML tools can support proactive food safety surveillance and risk mitigation** in agricultural and food production systems.

This repository is designed to meet competition requirements for:
- Model performance benchmarking  
- Interpretability and actionable insights  
- Code quality and reproducibility  

---

## 📖 Data Source and Citation

The dataset used in this competition submission is derived from the following peer-reviewed publication:

> **Liao, J., Guo, X., Weller, D. L., et al. (2021).**  
> *Nationwide genomic atlas of soil-dwelling Listeria reveals effects of selection and population ecology on pangenome evolution.*  
> **Nature Microbiology, 6, 1021–1030.**  
> https://doi.org/10.1038/s41564-021-00935-7

📌 **Please cite this paper when using the dataset or reporting results derived from this repository.**

---

## 🧪 Sample Analysis

### Competition Context

This project is developed as part of the **IAFP AI Benchmarking Student Competition**, which challenges participants to build and document predictive ML/DL models for real-world food safety problems using curated academic and industry datasets.

The specific focus of this submission is:
- **GIS-based pathogen presence prediction**
- **Environmental monitoring for food safety risk assessment**

---

## 🎯 Prediction Task

### Task Type
- **Binary classification**

### Objective
- Predict the **presence or absence of *Listeria spp.* in U.S. soil samples**

### Outcome Variable
- A binary presence/absence label derived from the number of *Listeria* isolates obtained per soil sample.

### Predictor Variables
- Soil physicochemical properties  
- Climate and moisture indicators  
- Land-use characteristics (e.g., cropland, shrubland)  
- Geographic information (latitude and longitude)

📄 A detailed description of all predictor and outcome variables is provided in:
- **`ListeriaSoil_Metadata.csv`**

📊 The cleaned and analysis-ready dataset used in this competition submission is:
- **`ListeriaSoil_clean.csv`**

---

## 📈 Evaluation Metrics

In accordance with the competition evaluation criteria, classification models developed using this pipeline are evaluated using:

- **ROC AUC**
- **Sensitivity (Recall)**
- **Specificity**
- **F1 Score**

Metric implementations are documented within the model training scripts (e.g., `Customize_script.py`) to ensure transparency and reproducibility.

---

## 🧰 Repository Structure

```text
IAFP_2026_Competition/
├── data/
│   ├── raw/            # Original datasets
│   └── processed/      # Analysis- and model-ready datasets
├── scripts/
│   ├── 01_univariate_screening.py
├── src/
│   ├── config.py       # Centralized path management
│   └── __init__.py
├── outputs/            # Figures and exported results
├── requirements.txt
└── README.md
```
---

### ⚙️ Installation
Dependencies

This project is implemented in Python and relies on the following core libraries:
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Statsmodels
- Matplotlib
- Seaborn
- TensorFlow / Keras (for downstream deep learning models)

All required dependencies are listed in requirements.txt.

---

### 🚀 Getting Started
1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/IAFP_2026_Competition.git
cd IAFP_2026_Competition
```
2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Run univariate screening (baseline analysis)
```bash
python -m scripts.01_univariate_screening --input ListeriaSoil_clean.csv
```
---

#### Optional flags:

- save-results : save outputs to data/processed/
- save-plots : export figures to outputs/
- show-plots : display figures interactively

---

#### 🔁 Reproducibility and Code Quality

Reproducibility is a core judging criterion for the competition. This repository ensures reproducible research by:

- Using repository-relative paths (no local or cloud-specific paths)
- Providing a fully script-based pipeline (no hidden notebook state)
- Recording metadata for each analysis run
- Using deterministic random seeds where applicable
- Maintaining modular, well-documented code

---

#### 🏆 Competition Information

This work is submitted to the IAFP AI Benchmarking Student Competition on Predictive Food Safety Models, supported by Cornell University, Agroknow, and academic–industry collaborators.

📄 Written report deadline: March 1, 2026
📍 Finalists presentation: IAFP 2026 Annual Meeting (New Orleans, LA)

---
## 🔍 Exploratory Analysis and Preprocessing Results

### Label Distribution

The final cleaned dataset includes **622 soil samples**, with the outcome variable defined as the binary presence or absence of *Listeria spp.* based on culture results.

```text
Listeria present:  311 samples (50%)
Listeria absent:   311 samples (50%)
```

### Correlation Structure of Predictors

A Pearson correlation matrix was computed across soil chemistry, climate, land-use, and geographic variables to examine linear dependencies and potential multicollinearity among predictors.

**Key observations:**
- Soil nutrients (e.g., total nitrogen, organic matter, and trace metals) form correlated clusters  
- Climate variables exhibit strong regional structure  
- Land-use variables show expected compositional relationships  
- No single predictor displays extreme correlation with the outcome label  

These patterns indicate structured but non-redundant predictors, supporting their retention in downstream tree-based models.

---

### Univariate Effect Size Analysis (Cliff’s Delta)

To quantify the magnitude and direction of associations between individual predictors and *Listeria* presence, **Cliff’s Delta** was used as a nonparametric effect size measure. This approach emphasizes practical relevance beyond statistical significance alone.

#### Top Variables by Absolute Cliff’s Delta

| Variable | Median (Absent) | Median (Present) | Δ (Present − Absent) | Cliff’s Δ | Adjusted p-value |
|---------|-----------------|------------------|----------------------|-----------|------------------|
| Shrubland (%) | 3.6009 | 0.3208 | −3.2801 | −0.44 | 5.9e−20 |
| Moisture | 0.1723 | 0.2887 | +0.1164 | +0.42 | 4.5e−18 |
| Cropland (%) | 1.1679 | 6.0231 | +4.8552 | +0.39 | 5.2e−16 |
| Longitude | −102.45 | −90.58 | +11.87 | +0.38 | 1.5e−15 |
| Pasture (%) | 0.3966 | 4.9012 | +4.5046 | +0.31 | 7.4e−11 |

All reported associations remained statistically significant after false discovery rate (FDR) correction.

---

### Land-use Composition by *Listeria* Presence

Comparisons of mean land-use composition revealed systematic differences between *Listeria*-positive and *Listeria*-negative soil samples.

**Observed patterns:**
- *Listeria*-absent soils are dominated by **shrubland and grassland**
- *Listeria*-present soils show higher proportions of **cropland and pasture**
- Shrubland coverage is markedly reduced in positive samples  

These patterns highlight the importance of land-use context in environmental *Listeria* persistence.

---

### Biological Interpretation

#### Land-use associations

*Listeria* presence is positively associated with **managed and disturbed landscapes**, particularly:
- Cropland  
- Pasture  

In contrast, *Listeria* absence is associated with **semi-natural vegetation**, including:
- Shrubland  
- Grassland  

This supports an **anthropogenic association hypothesis**, where *Listeria* persistence in soils reflects:
- Agricultural disturbance  
- Livestock activity  
- Manure application  
- Repeated environmental re-inoculation  

#### Soil and climate context

- Higher soil moisture and nutrient availability (e.g., nitrogen, organic matter, trace metals) are associated with *Listeria* presence  
- These variables likely act as **contextual modifiers**, influencing survival and persistence rather than serving as single causal drivers  

---

### Summary Statement

> *Univariate effect size analysis revealed that Listeria presence was negatively associated with semi-natural land cover (e.g., shrubland), while positively associated with soil moisture and managed land-use types, particularly cropland and pasture. However, substantial overlap in predictor distributions between presence and absence groups indicates that Listeria occurrence is not driven by single-variable thresholds, but rather by multivariate and nonlinear interactions, justifying the use of tree-based ensemble models for predictive modeling.*
--- 
<img width="2306" height="1238" alt="image" src="https://github.com/user-attachments/assets/cc47b825-cea6-486b-a40d-c3033fd27158" />

##### **Figure 1. Spatial distribution of Listeria soil samples across the United States.**
Geographic distribution of soil sampling locations included in the study. Each point represents an individual soil sample, with color indicating Listeria presence (1) or absence (0). State boundaries are shown for reference. Sampling locations are unevenly distributed across the continental United States, reflecting differences in regional sampling intensity.

The spatial distribution suggests that Listeria presence varies geographically, with clusters of positive samples in certain regions. These patterns may reflect regional differences in environmental conditions, land use, agricultural practices, or climate, rather than random occurrence, although uneven sampling density limits causal interpretation.

---
<img width="2306" height="1238" alt="Rplot55" src="https://github.com/user-attachments/assets/84f5eba6-67cb-4dc3-bd9b-788cd56e4fd9" />

#####  **Figure 2. Spatial distribution of Listeria soil samples incorporating elevation.**
Spatial distribution of soil samples across the continental United States, with point color indicating Listeria presence (1) or absence (0) and point size proportional to sampling site elevation (m). State boundaries are shown for geographic context. This visualization highlights heterogeneity in both sampling elevation and Listeria occurrence across regions.

The incorporation of elevation suggests a potential association between Listeria absence and higher-elevation sampling sites, as several negative samples (label = 0) are observed at comparatively higher elevations. While positive samples are present across a range of elevations, this visual pattern indicates that elevation—or correlated environmental factors such as temperature, moisture availability, and soil characteristics—may influence Listeria persistence. However, this observation is qualitative and warrants formal statistical evaluation.

<img width="2306" height="1238" alt="image" src="https://github.com/user-attachments/assets/e0fb4600-2640-4b32-8d81-65b6a7ec9287" />

##### **Figure 3. State-level prevalence of Listeria presence.**
State-level prevalence of Listeria presence estimated from soil samples aggregated by U.S. state. Prevalence is calculated as the proportion of samples positive for Listeria within each state. States with no available samples are shown in grey. Differences in prevalence should be interpreted with caution due to variability in sample size across states.

Variation in state-level prevalence suggests that broader regional factors, including climatic conditions, dominant land use, and agricultural intensity, may influence Listeria occurrence. Nonetheless, differences in sampling effort among states introduce uncertainty, and observed prevalence patterns likely reflect a combination of environmental drivers and data availability.

---
**Table 2.** State-level prevalence and concentration estimates of Listeria contamination in lettuce. For each state, the total number of samples (n), number of positives (k), estimated prevalence with 95% confidence intervals, and modeled mean and 95th percentile concentrations (Cs_Li_mean and Cs_Li_q95) are shown.

| State            | n   | k   | Prevalence | CI (95%)            | Cs_Li_mean | Cs_Li_q95 |
|------------------|-----|-----|------------|---------------------|------------|-----------|
| Kentucky         | 5   | 5   | 1.000      | [0.511, 1.000]      | 5.52E-01   | 5.53E-01  |
| South Carolina   | 5   | 5   | 1.000      | [0.511, 1.000]      | 5.52E-01   | 5.53E-01  |
| Indiana          | 26  | 23  | 0.885      | [0.702, 0.968]      | 1.44E-01   | 1.87E-01  |
| Louisiana        | 15  | 13  | 0.867      | [0.609, 0.975]      | 1.56E-01   | 2.07E-01  |
| Delaware         | 5   | 4   | 0.800      | [0.360, 0.980]      | 1.66E-01   | 2.24E-01  |
| Alabama          | 23  | 18  | 0.783      | [0.577, 0.908]      | 9.73E-02   | 1.20E-01  |
| New York         | 12  | 9   | 0.750      | [0.462, 0.917]      | 1.02E-01   | 1.26E-01  |
| Iowa             | 19  | 14  | 0.737      | [0.509, 0.885]      | 8.82E-02   | 1.08E-01  |
| Minnesota        | 44  | 32  | 0.727      | [0.580, 0.838]      | 7.38E-02   | 8.98E-02  |
| Ohio             | 22  | 16  | 0.727      | [0.516, 0.871]      | 8.34E-02   | 1.02E-01  |
| Illinois         | 20  | 14  | 0.700      | [0.479, 0.857]      | 7.91E-02   | 9.65E-02  |
| Maryland         | 10  | 7   | 0.700      | [0.392, 0.897]      | 9.25E-02   | 1.14E-01  |
| Pennsylvania     | 19  | 13  | 0.684      | [0.458, 0.848]      | 7.66E-02   | 9.34E-02  |
| North Dakota     | 30  | 20  | 0.667      | [0.487, 0.809]      | 6.69E-02   | 8.13E-02  |
| Arkansas         | 14  | 9   | 0.643      | [0.386, 0.838]      | 7.39E-02   | 9.00E-02  |
| Mississippi      | 5   | 3   | 0.600      | [0.229, 0.884]      | 8.77E-02   | 1.07E-01  |
| Georgia          | 19  | 11  | 0.579      | [0.362, 0.769]      | 5.93E-02   | 7.19E-02  |
| North Carolina   | 14  | 8   | 0.571      | [0.326, 0.787]      | 6.25E-02   | 7.59E-02  |
| Texas            | 45  | 24  | 0.533      | [0.391, 0.671]      | 4.48E-02   | 5.47E-02  |
| South Dakota     | 36  | 19  | 0.528      | [0.370, 0.680]      | 4.60E-02   | 5.62E-02  |
| Kansas           | 2   | 1   | 0.500      | [0.095, 0.905]      | 9.62E-02   | 1.18E-01  |
| Missouri         | 4   | 2   | 0.500      | [0.150, 0.850]      | 7.70E-02   | 9.38E-02  |
| Wisconsin        | 2   | 1   | 0.500      | [0.095, 0.905]      | 9.63E-02   | 1.19E-01  |
| Wyoming          | 9   | 4   | 0.444      | [0.188, 0.734]      | 5.35E-02   | 6.50E-02  |
| Michigan         | 14  | 6   | 0.429      | [0.213, 0.674]      | 4.53E-02   | 5.53E-02  |
| Tennessee        | 14  | 6   | 0.429      | [0.213, 0.674]      | 4.53E-02   | 5.53E-02  |
| Connecticut      | 3   | 1   | 0.333      | [0.056, 0.798]      | 6.47E-02   | 7.86E-02  |
| Massachusetts    | 3   | 1   | 0.333      | [0.056, 0.798]      | 6.46E-02   | 7.84E-02  |
| Montana          | 24  | 7   | 0.292      | [0.147, 0.494]      | 2.74E-02   | 3.43E-02  |
| Florida          | 11  | 3   | 0.273      | [0.092, 0.571]      | 3.41E-02   | 4.22E-02  |
| Oklahoma         | 9   | 2   | 0.222      | [0.053, 0.557]      | 3.29E-02   | 4.07E-02  |
| Utah             | 14  | 2   | 0.143      | [0.028, 0.412]      | 2.14E-02   | 2.72E-02  |
| Oregon           | 17  | 2   | 0.118      | [0.020, 0.356]      | 1.77E-02   | 2.29E-02  |
| Washington       | 9   | 1   | 0.111      | [0.000, 0.457]      | 2.46E-02   | 3.09E-02  |
| Maine            | 12  | 1   | 0.083      | [0.000, 0.375]      | 1.89E-02   | 2.44E-02  |
| New Mexico       | 17  | 1   | 0.059      | [0.000, 0.289]      | 1.37E-02   | 1.82E-02  |
| Colorado         | 18  | 1   | 0.056      | [0.000, 0.276]      | 1.30E-02   | 1.73E-02  |
| California       | 25  | 1   | 0.040      | [0.000, 0.211]      | 9.55E-03   | 1.32E-02  |
| Arizona          | 8   | 0   | 0.000      | [0.000, 0.372]      | 1.87E-02   | 2.41E-02  |
| Idaho            | 12  | 0   | 0.000      | [0.000, 0.282]      | 1.33E-02   | 1.77E-02  |
| Nevada           | 3   | 0   | 0.000      | [0.000, 0.617]      | 3.88E-02   | 4.76E-02  |
| New Jersey       | 3   | 0   | 0.000      | [0.000, 0.617]      | 3.88E-02   | 4.76E-02  |
| **Total**        | 621 | 310 | 0.4992     | –                   | 0.061      | 0.069     |
|                  |     |     |            |                     | -1.218     | -1.160    |

--- 

### Table description (statistical modeling details)

This table presents state-level estimates of *Listeria* prevalence and concentration derived using a probabilistic modeling framework that explicitly accounts for sampling uncertainty and upper-tail behavior relevant for exposure assessment.

#### Prevalence estimation

Prevalence for each state was estimated from the number of positive samples (*k*) out of the total number of samples analyzed (*n*). Uncertainty in prevalence was quantified using the **Agresti–Coull (AC) confidence interval**, defined as follows:

**Adjusted prevalence estimate:**

$$
\tilde{p} = \frac{k + z^2 / 2}{n + z^2}
$$

**95% Agresti–Coull confidence interval:**

$$
\text{CI}_{AC} = \tilde{p} \pm z \sqrt{\frac{\tilde{p}(1 - \tilde{p})}{n + z^2}}
$$

where \( z = 1.96 \) for a 95% confidence level.

To conservatively propagate prevalence uncertainty into the risk model, the upper bound of the 95% Agresti–Coull confidence interval was used to parameterize a Beta distribution representing state-level prevalence uncertainty:

$$
p_{\text{state}} \sim \text{Beta}(\alpha, \beta)
$$

where the Beta parameters were selected to be consistent with the observed data and the upper confidence limit.

#### Concentration modeling

Concentration estimates were derived using a Poisson-based modeling approach, assuming that the number of *Listeria* cells per gram follows a Poisson process conditional on contamination:

$$
C_{L_i} \mid p_{\text{state}} \sim \text{Poisson}(\lambda_{L_i})
$$

where ( $\lambda$) represents the mean concentration of *Listeria* (CFU/g).

From the resulting concentration distributions, two summary statistics were extracted:

- **Cs<sub>Lᵢ</sub><sup>mean</sup>**: mean modeled concentration (CFU/g)
- **Cs<sub>Lᵢ</sub><sup>q95</sup>**: 95th percentile of the modeled concentration distribution

These metrics together characterize both central tendency and upper-tail behavior relevant to exposure and risk estimation.

#### Pooled estimates

The final row of the table reports pooled estimates across all states based on the aggregated dataset.
