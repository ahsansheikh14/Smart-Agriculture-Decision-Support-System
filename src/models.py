"""
Train and evaluate ML models.
Phase 3: Decision Tree | Phase 4: KMeans | Phase 5: Linear Regression
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    silhouette_score,
)

# Allow import of preprocessing.py from the same src/ folder
sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import prepare_data, FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def train_decision_tree():
    """Train Decision Tree to predict crop from soil + weather features."""
    print("=== Phase 3: Decision Tree (Crop Recommendation) ===\n")

    # 1) Get prepared data from preprocessing.py
    data = prepare_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_crop_train"]
    y_test = data["y_crop_test"]

    # 2) Create and train the model
    # max_depth limits tree size so it does not overfit too much
    crop_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    crop_model.fit(X_train, y_train)  # <-- THIS is training

    # 3) Predict on test data (rows the model never saw during training)
    y_pred = crop_model.predict(X_test)

    # 4) Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}  (weighted average)")
    print(f"Recall   : {recall:.4f}  (weighted average)")
    print("\nDetailed report (per crop):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 5) Feature importance plot
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RESULTS_DIR / "feature_importance.png"

    importances = crop_model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.bar(FEATURE_COLUMNS, importances, color="seagreen")
    plt.title("Decision Tree - Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"\nFeature importance plot saved to: {plot_path}")

    return crop_model


def train_kmeans(n_clusters=4):
    """
    Train KMeans to group similar soil/climate conditions.

    Uses scaled features (X_scaled) from preprocessing.py.
    Does NOT use crop labels — unsupervised learning.
    """
    print("=== Phase 4: KMeans (Soil / Climate Clustering) ===\n")

    data = prepare_data()
    X = data["X"]                    # original values (for readable plot)
    X_scaled = data["X_scaled"]      # scaled values (for training)

    # 1) Create clusters (groups)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)  # <-- training + assign cluster IDs

    # 2) Silhouette score: how well-separated clusters are (-1 to 1, higher is better)
    sil_score = silhouette_score(X_scaled, clusters)
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette score : {sil_score:.4f}")
    print("  (closer to 1.0 means clearer, well-separated groups)")

    # Show how many farms fell in each cluster
    print("\nRows per cluster:")
    for i in range(n_clusters):
        count = (clusters == i).sum()
        print(f"  Cluster {i}: {count} rows")

    # 3) Scatter plot (2 features for easy visualization)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = RESULTS_DIR / "cluster_scatter.png"

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X["N"],
        X["K"],
        c=clusters,
        cmap="viridis",
        alpha=0.6,
        edgecolors="k",
        linewidths=0.2,
    )
    plt.title(f"KMeans Clusters (N vs K) — silhouette = {sil_score:.3f}")
    plt.xlabel("Nitrogen (N)")
    plt.ylabel("Potassium (K)")
    plt.colorbar(label="Cluster ID")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"\nCluster scatter plot saved to: {plot_path}")

    return kmeans


if __name__ == "__main__":
    train_kmeans()
