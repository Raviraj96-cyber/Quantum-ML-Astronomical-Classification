"""
Quantum Machine Learning for Astronomical Object Classification: An Ablation Study
Reconstructed from the project report (SDSS DR17, VQC vs RF/SVM, experiments E1-E6).

Usage:
    python qml_astro_classification.py --csv SDSS_DR17.csv              # everything
    python qml_astro_classification.py --csv SDSS_DR17.csv --only E5    # one experiment
    python qml_astro_classification.py --skip-vqc                       # only data + baselines
    python qml_astro_classification.py --plots-only                      # rebuild charts from JSON

Install (StatevectorSampler needs Qiskit 1.x, not 0.45):
    pip install "qiskit>=1.0" "qiskit-machine-learning>=0.8" scikit-learn numpy pandas matplotlib seaborn
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

SEED = 42
SAMPLES_PER_CLASS = 700
CLASS_MAP = {"STAR": 0, "GALAXY": 1, "QSO": 2}
CLASS_NAMES = ["Star", "Galaxy", "QSO"]
ALL_FEATURES = ["u", "g", "r", "i", "z", "redshift"]
OUT_DIR = "outputs"
DATA_DIR = "data"

# Ablation setups: name -> feature subset (number of qubits = number of features)
EXPERIMENTS = {
    "E1": ["u", "g", "r", "i", "z", "redshift"],  # 6 qubits
    "E2": ["redshift"],                           # 1 qubit (ZFeatureMap)
    "E3": ["u", "g", "r", "i", "z"],              # 5 qubits (no redshift)
    "E4": ["r", "i", "z", "redshift"],            # 4 qubits
    "E5": ["g", "r", "redshift"],                 # 3 qubits (best)
    "E6": ["r", "redshift"],                      # 2 qubits
}


# --------------------------------------------------------------------------
# Module 1: data ingestion, balancing, split, scaling
# --------------------------------------------------------------------------
def prepare_data(csv_path):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)
    print("Loaded:", df.shape)
    print(df["class"].value_counts())

    df = df[ALL_FEATURES + ["class"]].copy()
    df["label"] = df["class"].str.upper().map(CLASS_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    np.random.seed(SEED)
    parts = []
    for lbl in range(3):
        cls_df = df[df["label"] == lbl]
        idx = np.random.choice(cls_df.index, SAMPLES_PER_CLASS, replace=False)
        parts.append(df.loc[idx])
    balanced = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

    X = balanced[ALL_FEATURES].values
    y = balanced["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)

    # Fit scaler on the training set only (no data leakage)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = np.clip(scaler.transform(X_test), 0, 1)  # keep test values inside [0,1]

    np.save(f"{DATA_DIR}/X_train_balanced.npy", X_train)
    np.save(f"{DATA_DIR}/X_test_balanced.npy", X_test)
    np.save(f"{DATA_DIR}/y_train_balanced.npy", y_train)
    np.save(f"{DATA_DIR}/y_test_balanced.npy", y_test)
    print("Train:", X_train.shape, "Test:", X_test.shape)
    return X_train, X_test, y_train, y_test


def load_saved_data():
    return (np.load(f"{DATA_DIR}/X_train_balanced.npy"),
            np.load(f"{DATA_DIR}/X_test_balanced.npy"),
            np.load(f"{DATA_DIR}/y_train_balanced.npy"),
            np.load(f"{DATA_DIR}/y_test_balanced.npy"))


# --------------------------------------------------------------------------
# Shared evaluation helper
# --------------------------------------------------------------------------
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    per_class = (cm.diagonal() / cm.sum(axis=1)).tolist()
    return {
        "accuracy": float(acc), "precision_macro": float(p),
        "recall_macro": float(r), "f1_macro": float(f),
        "per_class_accuracy": dict(zip(CLASS_NAMES, per_class)),
        "confusion_matrix": cm.tolist(),
    }


# --------------------------------------------------------------------------
# Module 2: classical baselines
# --------------------------------------------------------------------------
def run_baselines(X_train, X_test, y_train, y_test):
    os.makedirs(OUT_DIR, exist_ok=True)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "SVM": SVC(kernel="rbf", C=1.0),
    }
    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        res = evaluate(y_test, model.predict(X_test))
        res["time_seconds"] = elapsed
        results[name] = res
        print(f"{name}: acc={res['accuracy']:.4f}  time={elapsed:.2f}s")
    with open(f"{OUT_DIR}/classical_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    return results


# --------------------------------------------------------------------------
# Modules 3-5: VQC construction, training, evaluation
# --------------------------------------------------------------------------
def run_vqc(exp_name, X_train, X_test, y_train, y_test, maxiter=150):
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, EfficientSU2
    from qiskit.primitives import StatevectorSampler
    from qiskit_machine_learning.algorithms.classifiers import VQC
    from qiskit_machine_learning.optimizers import COBYLA

    os.makedirs(OUT_DIR, exist_ok=True)
    feats = EXPERIMENTS[exp_name]
    cols = [ALL_FEATURES.index(f) for f in feats]
    n = len(feats)
    Xtr, Xte = X_train[:, cols], X_test[:, cols]

    # ZZFeatureMap needs >= 2 qubits, so ZFeatureMap is used for the 1-qubit case (E2)
    if n >= 2:
        feature_map = ZZFeatureMap(feature_dimension=n, reps=2)
    else:
        feature_map = ZFeatureMap(feature_dimension=n, reps=2)
    ansatz = EfficientSU2(n, reps=2, entanglement="linear")

    loss_history = []

    def callback(weights, obj_value):
        loss_history.append(float(obj_value))
        if len(loss_history) % 10 == 0:
            print(f"  [{exp_name}] iter {len(loss_history):3d}  loss={obj_value:.4f}")

    np.random.seed(SEED)
    vqc = VQC(
        sampler=StatevectorSampler(seed=SEED),
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=maxiter),
        callback=callback,
    )

    print(f"\n=== {exp_name}: features={feats}, qubits={n} ===")
    t0 = time.time()
    vqc.fit(Xtr, y_train)
    minutes = (time.time() - t0) / 60

    y_pred = vqc.predict(Xte)
    # VQC may return one-hot predictions; convert to class indices
    y_pred = np.argmax(y_pred, axis=1) if np.ndim(y_pred) == 2 else y_pred

    res = evaluate(y_test, y_pred)
    res.update({"experiment": exp_name, "features": feats, "qubits": n,
                "training_time_minutes": minutes, "loss_history": loss_history})
    print(f"{exp_name}: acc={res['accuracy']:.4f}  time={minutes:.1f} min")

    with open(f"{OUT_DIR}/vqc_{exp_name.lower()}_results.json", "w") as fp:
        json.dump(res, fp, indent=2)

    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, color="#5b4bbf")
    plt.xlabel("COBYLA iteration")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{exp_name} loss curve ({n} qubits: {', '.join(feats)})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/vqc_{exp_name.lower()}_loss_curve.png", dpi=150)
    plt.close()
    return res


# --------------------------------------------------------------------------
# Module 7: visualisation
# --------------------------------------------------------------------------
def make_plots():
    with open(f"{OUT_DIR}/classical_results.json") as fp:
        classical = json.load(fp)
    vqc = {}
    for e in EXPERIMENTS:
        path = f"{OUT_DIR}/vqc_{e.lower()}_results.json"
        if os.path.exists(path):
            with open(path) as fp:
                vqc[e] = json.load(fp)
    if not vqc:
        print("No VQC results found; skipping plots.")
        return

    best = max(vqc, key=lambda k: vqc[k]["accuracy"])

    # Fig 8.1 accuracy comparison
    names = ["RF", "SVM"] + list(vqc)
    accs = [classical["Random Forest"]["accuracy"] * 100,
            classical["SVM"]["accuracy"] * 100] + [vqc[e]["accuracy"] * 100 for e in vqc]
    colors = ["#1a5fa6"] * 2 + ["#2e6b10" if e == best else "#636059" for e in vqc]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, accs, color=colors)
    plt.axhline(100 / 3, color="r", ls="--", lw=0.8, label="Random baseline (33.33%)")
    for b, a in zip(bars, accs):
        plt.text(b.get_x() + b.get_width() / 2, a + 1, f"{a:.2f}%", ha="center", fontsize=8)
    plt.ylabel("Accuracy (%)")
    plt.title("Figure 8.1 - Accuracy Comparison: Classical vs VQC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig8_1_accuracy_comparison.png", dpi=150)
    plt.close()

    # Fig 8.2 accuracy vs qubits
    order = sorted(vqc, key=lambda k: vqc[k]["qubits"])
    plt.figure(figsize=(8, 5))
    plt.plot([f"{vqc[e]['qubits']}q\n({e})" for e in order],
             [vqc[e]["accuracy"] * 100 for e in order], "o-", color="#5b4bbf")
    plt.axhline(100 / 3, color="r", ls="--", lw=0.8)
    plt.ylabel("Accuracy (%)")
    plt.title("Figure 8.2 - VQC Accuracy vs Number of Qubits")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig8_2_accuracy_vs_qubits.png", dpi=150)
    plt.close()

    # Fig 8.3 training time
    plt.figure(figsize=(8, 5))
    times = [vqc[e]["training_time_minutes"] for e in vqc]
    plt.bar(list(vqc), times, color=["#a52a2a" if t == max(times) else "#636059" for t in times])
    plt.ylabel("Training time (minutes)")
    plt.title("Figure 8.3 - Training Time Comparison")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig8_3_training_time.png", dpi=150)
    plt.close()

    # Fig 8.4 confusion matrix of best model
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(vqc[best]["confusion_matrix"]), annot=True, fmt="d", cmap="YlGn",
                xticklabels=[f"Pred: {c}" for c in CLASS_NAMES],
                yticklabels=[f"Actual: {c}" for c in CLASS_NAMES])
    plt.title(f"Figure 8.4 - Confusion Matrix: {best} (Best VQC)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig8_4_confusion_matrix_best.png", dpi=150)
    plt.close()

    # Fig 8.5 ranked performance
    ranked = sorted(vqc, key=lambda k: vqc[k]["accuracy"])
    plt.figure(figsize=(8, 5))
    plt.barh([f"{e} ({vqc[e]['qubits']}q)" for e in ranked],
             [vqc[e]["accuracy"] * 100 for e in ranked], color="#20a07a")
    plt.xlabel("Accuracy (%)")
    plt.title("Figure 8.5 - Ranked VQC Performance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig8_5_ranked_performance.png", dpi=150)
    plt.close()

    # Summary table
    print("\nSession  Qubits  Acc     Prec    Recall  F1      Time(min)")
    for e in vqc:
        r = vqc[e]
        print(f"{e:7s}  {r['qubits']:6d}  {r['accuracy']*100:5.2f}%  {r['precision_macro']*100:5.2f}%  "
              f"{r['recall_macro']*100:5.2f}%  {r['f1_macro']*100:5.2f}%  {r['training_time_minutes']:.1f}")
    print("Plots saved in", OUT_DIR)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Path to SDSS DR17 CSV (needs u,g,r,i,z,redshift,class)")
    ap.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), help="Run only these experiments")
    ap.add_argument("--skip-vqc", action="store_true")
    ap.add_argument("--plots-only", action="store_true")
    ap.add_argument("--maxiter", type=int, default=150)
    args = ap.parse_args()

    if args.plots_only:
        make_plots()
        return

    if args.csv:
        X_train, X_test, y_train, y_test = prepare_data(args.csv)
    else:
        X_train, X_test, y_train, y_test = load_saved_data()

    run_baselines(X_train, X_test, y_train, y_test)

    if not args.skip_vqc:
        for exp in (args.only or list(EXPERIMENTS)):
            run_vqc(exp, X_train, X_test, y_train, y_test, maxiter=args.maxiter)

    make_plots()


if __name__ == "__main__":
    main()
