# Assignment 3 — Measuring Associations and Statistical Significance
**Course:** CSDS 413 – Introduction to Data Analysis  
**Instructor:** Dr. Mehmet Koyutürk  
**Semester:** Fall 2025  

---

## Overview
This assignment explores the quantification of associations between variables—both binary and continuous—along with the assessment of their statistical significance and correction for multiple hypothesis testing.  
Students will combine computational experimentation with statistical reasoning to interpret relationships among variables in real datasets.

---

## Task 1 – Associations Between Binary Variables
**Goal:** Measure and evaluate pairwise associations between genomic variants or binary attributes.

### Methods
- **Mutual Information (MI)** – information‐theoretic dependence measure  
- **Jaccard Index** – overlap‐based similarity  
- **Pearson’s Chi-Squared Test** – contingency‐based statistical association  

### Significance Assessment
- **Permutation Testing** (using resampling from empirical distributions)  
- **Parametric Testing** (using analytical approximations when applicable)

### Multiple Hypothesis Correction
- **Bonferroni Correction (FWER control)**  
- **Benjamini–Hochberg Procedure (FDR control)**  

---

## Task 2 – Correlations Between Continuous Variables
**Goal:** Evaluate linear associations between continuous variables.

### Methods
- **Pearson Correlation Coefficient** – measures linear dependence  
- **Permutation or Parametric Tests** – to assess significance of r  
- **Comparative Visualization**
  - Scatter plots with trend lines  
  - Histograms or density plots of correlation values  
  - Annotated tables of correlation coefficients and p-values  

---

## Deliverables
1. **Report (in LaTeX preferred)**  
   - Introduction and methodology  
   - Results with figures and tables  
   - Interpretation and discussion of significance and corrections  
2. **Code and Data**  
   - Clearly commented Python or R scripts  
   - Reproducible pipeline for computing statistics and plots  

---

## Expected Learning Outcomes
- Understand how to quantify association between variables of different types.  
- Apply and interpret permutation and parametric significance tests.  
- Correct for multiple hypothesis testing to avoid false discoveries.  
- Visualize and critically interpret statistical relationships in data.  


