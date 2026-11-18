# Quantum Machine Learning for Astronomical Object Classification: An Ablation Study

A college team mini-project exploring Variational Quantum Classifiers (VQC) for classifying astronomical objects using photometric data from the Sloan Digital Sky Survey Data Release 17 (SDSS DR17).

## Team

- Aditya Balasaheb Gaikwad
- Sarthak Vilas Patil
- Raviraj Ambadas Babar

**Project guide:** Prof. G. G. Shingan  
**Department:** Computer Science & Engineering  
**Institute:** K.E. Society's Rajarambapu Institute of Technology, Rajaramnagar  
**Academic year:** 2025–2026

## Project overview

The project studies the classification of three astronomical object classes:

- Stars
- Galaxies
- Quasi-Stellar Objects (QSOs)

The report describes a balanced dataset of 2,100 samples (700 per class), selected from SDSS DR17 data. Six photometric/redshift features are considered: `u`, `g`, `r`, `i`, `z`, and `redshift`.

The work compares two classical machine-learning baselines with a quantum-classical Variational Quantum Classifier (VQC). It also performs six ablation experiments (E1–E6) to study how feature selection and qubit count affect performance.

## Methodology

1. Load and prepare SDSS DR17 data.
2. Balance the data to 700 samples per class.
3. Split the data using a stratified 80:20 train-test split.
4. Fit `MinMaxScaler` on the training set and apply it to the test set.
5. Train classical baselines:
   - Random Forest (100 estimators)
   - Support Vector Machine (RBF kernel)
6. Build and train VQC experiments using:
   - Qiskit `ZZFeatureMap` (with `ZFeatureMap` for the one-feature experiment)
   - `EfficientSU2` ansatz
   - COBYLA optimizer
   - Statevector simulation
7. Evaluate accuracy, precision, recall, F1-score, confusion matrices, and training time.

## Reported results

| Model / experiment | Features | Qubits | Accuracy |
|---|---|---:|---:|
| Random Forest | Classical baseline | — | 97.38% |
| SVM (RBF) | Classical baseline | — | 91.67% |
| VQC E1 | u, g, r, i, z, redshift | 6 | 56.43% |
| VQC E2 | redshift | 1 | 49.05% |
| VQC E3 | u, g, r, i, z | 5 | 52.86% |
| VQC E4 | r, i, z, redshift | 4 | 66.43% |
| VQC E5 | g, r, redshift | 3 | 71.19% |
| VQC E6 | r, redshift | 2 | 64.05% |

According to the report, E5 achieved the highest VQC accuracy (71.19%). The Random Forest baseline achieved 97.38%, outperforming the VQC configurations in these experiments.

## Technology stack

- Python 3.9 or 3.10
- Qiskit 1.x
- qiskit-machine-learning 0.8.x
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

The supplied Python script uses `StatevectorSampler`, so its Qiskit dependencies are newer than the versions listed in the original report.

## Repository structure

```text
Quantum-ML-Mini-Project/
├── README.md
├── docs/
│   └── Mini_Project_Report.pdf
├── src/
│   └── qml_astro_classification.py
├── notebooks/        # Add Jupyter notebooks here, if used
├── results/          # Add generated plots, metrics, and confusion matrices
└── requirements.txt
```

## Setup

Create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

**Code note:** The supplied script is described in its header as reconstructed from the project report. It is included as `src/qml_astro_classification.py`; confirm with your teammates that this is the final team-approved implementation before publishing.

**Dataset note:** The SDSS CSV dataset is not included. The script expects a CSV with columns `u`, `g`, `r`, `i`, `z`, `redshift`, and `class`, where `class` contains `STAR`, `GALAXY`, and `QSO`. It samples 700 rows from each class, so the input must contain at least 700 usable rows per class.

Run from the repository root:
```bash
python src/qml_astro_classification.py --csv data/SDSS_DR17.csv
```
To run only E5:
```bash
python src/qml_astro_classification.py --csv data/SDSS_DR17.csv --only E5
```
To run classical baselines only:
```bash
python src/qml_astro_classification.py --csv data/SDSS_DR17.csv --skip-vqc
```
The script writes processed arrays under `data/` and metrics/plots under `outputs/`. Create the `data/` folder and place the permitted CSV there. Do not upload restricted or personally identifying data.

## Scope and limitations

The report states that experiments were run using a classical statevector simulator. Execution on real quantum hardware, large-scale processing of the full SDSS catalog, and deployment as a live web service were outside the project's scope.

## Academic attribution

This is a collaborative college mini-project. Keep the team attribution intact. Before making the repository public, confirm that all team members and the project guide are comfortable with public sharing, and check that the dataset and any included materials may be redistributed.

## Project report

The submitted mini-project report is available at [`docs/Mini_Project_Report.pdf`](docs/Mini_Project_Report.pdf).
