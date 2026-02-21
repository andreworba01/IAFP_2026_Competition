# Listeria in Soil — IAFP AI Benchmarking Student Competition 🏆🦠

🥬 This repository contains an end-to-end, reproducible **machine learning pipeline** developed for the **IAFP AI Benchmarking Student Competition on Predictive Food Safety Models**. The project addresses the challenge theme:

> **Predicting pathogen presence in food production environments**

Using a nationwide GIS-based dataset of U.S. soil samples, this work focuses on predicting the **presence or absence of *Listeria spp.* in soil** based on environmental, soil, climate, land-use, and geographic variables. The goal is to demonstrate how **AI/ML tools can support proactive food safety surveillance and risk mitigation** in agricultural and food production systems.

This repository is designed to meet competition requirements for:
- Model performance benchmarking  
- Interpretability and actionable insights  
- Code quality and reproducibility  

---

## Data Source and Citation

The dataset used in this competition submission is derived from the following peer-reviewed publication:

> **Liao, J., Guo, X., Weller, D. L., et al. (2021).**  
> *Nationwide genomic atlas of soil-dwelling Listeria reveals effects of selection and population ecology on pangenome evolution.*  
> **Nature Microbiology, 6, 1021–1030.**  
> https://doi.org/10.1038/s41564-021-00935-7

📌 **Please cite this paper when using the dataset or reporting results derived from this repository.**

---

## Sample Analysis

### Competition Context

This project is developed as part of the **IAFP AI Benchmarking Student Competition**, which challenges participants to build and document predictive ML/DL models for real-world food safety problems using curated academic and industry datasets.

The specific focus of this submission is:
- **GIS-based pathogen presence prediction**
- **Environmental monitoring for food safety risk assessment**

---

## Prediction Task

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

A detailed description of all predictor and outcome variables is provided in:
- **`ListeriaSoil_Metadata.csv`**

📊 The cleaned and analysis-ready dataset used in this competition submission is:
- **`ListeriaSoil_clean.csv`**

---

## valuation Metrics

In accordance with the competition evaluation criteria, classification models developed using this pipeline are evaluated using:

- **ROC AUC**
- **Sensitivity (Recall)**
- **Specificity**
- **F1 Score**

Metric implementations are documented within the model training scripts (e.g., `Customize_script.py`) to ensure transparency and reproducibility.

---

## Repository Structure

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

###  Getting Started
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

#### Reproducibility and Code Quality

Reproducibility is a core judging criterion for the competition. This repository ensures reproducible research by:

- Using repository-relative paths (no local or cloud-specific paths)
- Providing a fully script-based pipeline (no hidden notebook state)
- Recording metadata for each analysis run
- Using deterministic random seeds where applicable
- Maintaining modular, well-documented code

---

#### Competition Information

This work is submitted to the IAFP AI Benchmarking Student Competition on Predictive Food Safety Models, supported by Cornell University, Agroknow, and academic–industry collaborators.

📄 Written report deadline: March 1, 2026
📍 Finalists presentation: IAFP 2026 Annual Meeting (New Orleans, LA)

---
## Exploratory Analysis and Preprocessing Results

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
>
---

## Model Selection (Food-Safety Priority)

We evaluated multiple binary classifiers (LogReg, SVM-RBF, Random Forest, ExtraTrees, HistGradientBoosting, LightGBM, and CatBoost) using an **80/20 train–test split** and **cross-validation** on the training set. Because this is a food-safety screening task, we prioritized **high sensitivity (recall)** to minimize **false negatives** (missed positives).

To reflect this priority, we:
- Computed **out-of-fold (OOF)** predicted probabilities on the training set for each model.
- Selected a **model-specific decision threshold** to achieve a target recall (e.g., **≥ 0.90**) based on OOF predictions.
- Ranked models primarily by **fewest FN**, then **fewest FP**, using **recall** as a tie-breaker.
- Fine-tuned the best-performing model(s) via hyperparameter search and evaluated final performance on the held-out test set.

### Ranked Model Results (Test Set)

| Model | Threshold | TN | FP | FN | TP | Sensitivity (Recall) | Specificity | Balanced Accuracy | ROC AUC | Avg Precision | Chosen Thr (from OOF) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LogReg (balanced) | 0.321854 | 42 | 21 | 2 | 60 | 0.967742 | 0.666667 | 0.817204 | 0.890169 | 0.833814 | 0.321854 |
| SVM-RBF (balanced) | 0.307001 | 41 | 22 | 4 | 58 | 0.935484 | 0.650794 | 0.793139 | 0.906298 | 0.906705 | 0.307001 |
| HistGB | 0.198421 | 56 | 7 | 5 | 57 | 0.919355 | 0.888889 | 0.904122 | 0.958013 | 0.962916 | 0.198421 |
| LightGBM (recall-leaning) | 0.261349 | 55 | 8 | 6 | 56 | 0.903226 | 0.873016 | 0.888121 | 0.960573 | 0.960990 | 0.261349 |
| ExtraTrees (balanced) | 0.469331 | 50 | 13 | 6 | 56 | 0.903226 | 0.793651 | 0.848438 | 0.920123 | 0.931038 | 0.469331 |
| CatBoost (recall-leaning) | 0.495447 | 58 | 5 | 7 | 55 | 0.887097 | 0.920635 | 0.903866 | 0.965438 | 0.970190 | 0.495447 |
| RF (balanced) | 0.457408 | 53 | 10 | 8 | 54 | 0.870968 | 0.841270 | 0.856119 | 0.940348 | 0.944795 | 0.457408 |


## Results
![AUC curve](outputs/AUC%20curve.png)
--- 
### Summary of Spatila distribution of Listeria in soil samples across the United States. 

---
<img width="2306" height="1238" alt="Rplot55" src="https://github.com/user-attachments/assets/84f5eba6-67cb-4dc3-bd9b-788cd56e4fd9" />

#####  **Figure 1. Spatial distribution of Listeria soil samples incorporating elevation.**
Spatial distribution of soil samples across the continental United States, with point color indicating Listeria presence (1) or absence (0) and point size proportional to sampling site elevation (m). State boundaries are shown for geographic context. This visualization highlights heterogeneity in both sampling elevation and Listeria occurrence across regions.

The incorporation of elevation suggests a potential association between Listeria absence and higher-elevation sampling sites, as several negative samples (label = 0) are observed at comparatively higher elevations. While positive samples are present across a range of elevations, this visual pattern indicates that elevation—or correlated environmental factors such as temperature, moisture availability, and soil characteristics—may influence Listeria persistence. However, this observation is qualitative and warrants formal statistical evaluation.

<img width="2306" height="1238" alt="image" src="https://github.com/user-attachments/assets/e0fb4600-2640-4b32-8d81-65b6a7ec9287" />

##### **Figure 2. State-level prevalence of Listeria presence.**
State-level prevalence of Listeria presence estimated from soil samples aggregated by U.S. state. Prevalence is calculated as the proportion of samples positive for Listeria within each state. States with no available samples are shown in grey. Differences in prevalence should be interpreted with caution due to variability in sample size across states.

Variation in state-level prevalence suggests that broader regional factors, including climatic conditions, dominant land use, and agricultural intensity, may influence Listeria occurrence. Nonetheless, differences in sampling effort among states introduce uncertainty, and observed prevalence patterns likely reflect a combination of environmental drivers and data availability.

---
**Table 2.** State-level prevalence and concentration estimates of Listeria contamination in lettuce. For each state, the total number of samples (n), number of positives (k), estimated prevalence with 95% confidence intervals, and modeled mean and 95th percentile concentrations (Cs_Li_mean and Cs_Li_q95) are shown.

| State          | n  | Estimated Prevalence (π̂) | pU    | CI (95%)         | Cs_Li_mean (CFU/g) | sim_q95 (CFU/g) |
|----------------|----|---------------------------|-------|------------------|-------------------|----------------|
| Kentucky        | 5  | 1.000 | 1.000 | [0.511, 1.000] | 9.818E-02 | 1.915E-01 |
| South Carolina  | 5  | 1.000 | 1.000 | [0.511, 1.000] | 9.786E-02 | 1.908E-01 |
| Indiana         | 26 | 0.885 | 0.968 | [0.702, 0.968] | 1.156E-01 | 1.723E-01 |
| Louisiana       | 15 | 0.867 | 0.975 | [0.609, 0.975] | 1.356E-01 | 2.300E-01 |
| Delaware        | 5  | 0.800 | 0.980 | [0.360, 0.980] | 9.763E-02 | 1.892E-01 |
| Alabama         | 23 | 0.783 | 0.908 | [0.577, 0.908] | 9.099E-02 | 1.341E-01 |
| New York        | 12 | 0.750 | 0.917 | [0.462, 0.917] | 8.722E-02 | 1.429E-01 |
| Iowa            | 19 | 0.737 | 0.885 | [0.509, 0.885] | 8.386E-02 | 1.264E-01 |
| Minnesota       | 44 | 0.727 | 0.838 | [0.580, 0.838] | 7.202E-02 | 9.550E-02 |
| Ohio            | 22 | 0.727 | 0.871 | [0.516, 0.871] | 7.589E-02 | 1.114E-01 |
| Illinois        | 20 | 0.700 | 0.857 | [0.479, 0.857] | 7.273E-02 | 1.080E-01 |
| Maryland        | 10 | 0.700 | 0.897 | [0.392, 0.897] | 8.080E-02 | 1.364E-01 |
| Pennsylvania    | 19 | 0.684 | 0.848 | [0.458, 0.848] | 7.064E-02 | 1.057E-01 |
| North Dakota    | 30 | 0.667 | 0.809 | [0.487, 0.809] | 6.309E-02 | 8.801E-02 |
| Arkansas        | 14 | 0.643 | 0.838 | [0.386, 0.838] | 7.267E-02 | 1.145E-01 |
| Mississippi     | 5  | 0.600 | 0.884 | [0.229, 0.884] | 5.793E-02 | 1.108E-01 |
| Georgia         | 19 | 0.579 | 0.769 | [0.362, 0.769] | 6.061E-02 | 9.043E-02 |
| North Carolina  | 14 | 0.571 | 0.787 | [0.326, 0.787] | 5.943E-02 | 9.350E-02 |
| Texas           | 45 | 0.533 | 0.671 | [0.391, 0.671] | 4.396E-02 | 5.852E-02 |
| South Dakota    | 36 | 0.528 | 0.680 | [0.370, 0.680] | 4.396E-02 | 6.029E-02 |
| Kansas          | 2  | 0.500 | 0.905 | [0.095, 0.905] | 7.343E-02 | 1.631E-01 |
| Missouri        | 4  | 0.500 | 0.850 | [0.150, 0.850] | 5.151E-02 | 1.034E-01 |
| Wisconsin       | 2  | 0.500 | 0.905 | [0.095, 0.905] | 7.331E-02 | 1.631E-01 |
| Wyoming         | 9  | 0.444 | 0.734 | [0.188, 0.734] | 5.713E-02 | 9.725E-02 |
| Michigan        | 14 | 0.429 | 0.674 | [0.213, 0.674] | 4.131E-02 | 6.612E-02 |
| Tennessee       | 14 | 0.429 | 0.674 | [0.213, 0.674] | 4.149E-02 | 6.647E-02 |
| Connecticut     | 3  | 0.333 | 0.798 | [0.056, 0.798] | 4.325E-02 | 9.283E-02 |
| Massachusetts   | 3  | 0.333 | 0.798 | [0.056, 0.798] | 4.328E-02 | 9.280E-02 |
| Montana         | 24 | 0.292 | 0.494 | [0.147, 0.494] | 2.850E-02 | 4.295E-02 |
| Florida         | 11 | 0.273 | 0.571 | [0.092, 0.571] | 3.284E-02 | 5.624E-02 |
| Oklahoma        | 9  | 0.222 | 0.557 | [0.053, 0.557] | 3.371E-02 | 5.982E-02 |
| Utah            | 14 | 0.143 | 0.412 | [0.028, 0.412] | 2.400E-02 | 4.094E-02 |
| Oregon          | 17 | 0.118 | 0.356 | [0.020, 0.356] | 1.899E-02 | 3.231E-02 |
| Washington      | 9  | 0.111 | 0.457 | [0.000, 0.457] | 2.577E-02 | 4.767E-02 |
| Maine           | 12 | 0.083 | 0.375 | [0.000, 0.375] | 2.352E-02 | 4.155E-02 |
| New Mexico      | 17 | 0.059 | 0.289 | [0.000, 0.289] | 1.565E-02 | 2.744E-02 |
| Colorado        | 18 | 0.056 | 0.276 | [0.000, 0.276] | 1.471E-02 | 2.572E-02 |
| California      | 25 | 0.040 | 0.211 | [0.000, 0.211] | 1.026E-02 | 1.801E-02 |
| Arizona         | 8  | 0.000 | 0.372 | [0.000, 0.372] | 2.183E-02 | 4.267E-02 |
| Idaho           | 12 | 0.000 | 0.282 | [0.000, 0.282] | 1.408E-02 | 2.739E-02 |
| Nevada          | 3  | 0.000 | 0.617 | [0.000, 0.617] | 4.338E-02 | 9.284E-02 |
| New Jersey      | 3  | 0.000 | 0.617 | [0.000, 0.617] | 4.331E-02 | 9.303E-02 |
| **Total**       | 621|  –     | –                      |    –   |   – |–
| **Average**    |     |0.461	|  0.705      |      –        | 5.609E-02	 |9.628E-02

   |


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

Estimating *Listeria* concentration in soil is challenging. Denis et al. (2022) reported concentrations of up to **6 MPN/g** in soils surrounding a lagoon in France. This value can therefore be considered an approximate **upper limit** that would be difficult to observe in non-agricultural soils, such as those sampled in the study used to derive the concentration estimates reported in Table 2.  

Furthermore, Locatelli et al. (2013) demonstrated that the **limit of detection (LOD)** in dry soil can be as high as **10⁴ CFU/g of dry soil**, which helps explain why relatively few studies are able to quantitatively estimate *Listeria* concentrations in soil matrices. <mark> Therefore, the approximation presented above provides a reasonable basis for risk assessment and modeling.</mark>

#### Pooled estimates

The final row of the table reports pooled estimates across all states based on the aggregated dataset.

--


### Predicted Listeria Risk Hotspots Near Rivers and Lakes

This map shows the model-predicted probability of Listeria presence at sampled sites, overlaid with nearby rivers and lakes. Higher-risk locations (e.g., ≥0.60) stand out as potential hotspots, helping farmers prioritize where to sample, focus water-related controls, and plan harvest timing to reduce contamination risk.

<img width="2306" height="1238" alt="Rplot55" src="outputs/Predicted_probability_presence_rivers_lakes.png" />

---


---

# REFERENCES
Bilder, C. R., & Loughin, T. M. (2014). Analysis of Categorical Data with R (p. 55).Chapman & Hall/CRC Texts in Statistical Science. CRC Press. Kindle Edition.

Denis, M., Ziebal, C., Boscher, E., Picard, S., Perrot, M., Nova, M. V., Roussel, S., Diara, A., & Pourcher, A. M. (2022). Occurrence and diversity of *Listeria monocytogenes* isolated from two pig manure treatment plants in France. *Microbes and Environments, 37*(4), ME22019. https://doi.org/10.1264/jsme2.ME22019  

Locatelli, A., Depret, G., Jolivet, C., Henry, S., Dequiedt, S., Piveteau, P., & Hartmann, A. (2013). Nation-wide study of the occurrence of *Listeria monocytogenes* in French soils using culture-based and molecular detection methods. *Journal of Microbiological Methods, 93*(3), 242–250. https://doi.org/10.1016/j.mimet.2013.03.017

