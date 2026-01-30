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

**Figure 1. Spatial distribution of Listeria soil samples across the United States.**
Geographic distribution of soil sampling locations included in the study. Each point represents an individual soil sample, with color indicating Listeria presence (1) or absence (0). State boundaries are shown for reference. Sampling locations are unevenly distributed across the continental United States, reflecting differences in regional sampling intensity.

---
<img width="2306" height="1238" alt="Rplot55" src="https://github.com/user-attachments/assets/84f5eba6-67cb-4dc3-bd9b-788cd56e4fd9" />
**Figure 2. Spatial distribution of Listeria soil samples incorporating elevation.**
Spatial distribution of soil samples across the continental United States, with point color indicating Listeria presence (1) or absence (0) and point size proportional to sampling site elevation (m). State boundaries are shown for geographic context. This visualization highlights heterogeneity in both sampling elevation and Listeria occurrence across regions.

<img width="2306" height="1238" alt="image" src="https://github.com/user-attachments/assets/e0fb4600-2640-4b32-8d81-65b6a7ec9287" />
**Figure 3. State-level prevalence of Listeria presence.**
State-level prevalence of Listeria presence estimated from soil samples aggregated by U.S. state. Prevalence is calculated as the proportion of samples positive for Listeria within each state. States with no available samples are shown in grey. Differences in prevalence should be interpreted with caution due to variability in sample size across states.


