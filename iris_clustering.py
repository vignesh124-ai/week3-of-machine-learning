"""
=============================================================================
Mini Project 3: Iris Flower Clustering Project
=============================================================================
Tasks Accomplished:
  1. Load and explore the Iris dataset (UCI / scikit-learn).
  2. Standardize features using StandardScaler.
  3. Apply PCA (Principal Component Analysis) for dimensionality reduction.
  4. Determine optimal k using the Elbow Method and Silhouette Analysis.
  5. Apply K-Means (k=3) clustering.
  6. Visualize clusters in 2D PCA space, 3D PCA space, and original feature space.
  7. Map and compare predicted clusters against ground truth labels.
  8. Compute unsupervised & supervised clustering evaluation metrics.
=============================================================================
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    confusion_matrix,
    classification_report,
)

# Output directory for saving plots and results
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.titlesize"] = 14
plt.rcParams["figure.titleweight"] = "bold"


def load_and_explore_data():
    """Loads the Iris dataset, converts it to DataFrame and prints summary statistics."""
    print("=" * 70)
    print("STEP 1: DATA LOADING AND EXPLORATION")
    print("=" * 70)
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Features: {iris.feature_names}")
    print(f"Target classes: {list(iris.target_names)}")
    print(f"Class distribution:\n{df['species'].value_counts().to_string()}\n")
    print("Feature summary statistics:")
    print(df[iris.feature_names].describe().round(2).to_string())
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())

    return iris, df


def scale_features(X):
    """Standardizes features to zero mean and unit variance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return scaler, X_scaled


def evaluate_optimal_k(X_scaled):
    """
    Evaluates inertia (Elbow Method) and silhouette score for k in range 1 to 10.
    Generates and saves the Elbow and Silhouette plot.
    """
    print("\n" + "=" * 70)
    print("STEP 2: FINDING OPTIMAL K (ELBOW METHOD & SILHOUETTE ANALYSIS)")
    print("=" * 70)

    k_range = range(1, 11)
    inertias = []
    silhouette_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        if k >= 2:
            score = silhouette_score(X_scaled, kmeans.labels_)
            silhouette_scores.append(score)
            print(f"k = {k:2d} | Inertia (WCSS) = {kmeans.inertia_:8.2f} | Silhouette Score = {score:.4f}")
        else:
            print(f"k = {k:2d} | Inertia (WCSS) = {kmeans.inertia_:8.2f} | Silhouette Score = N/A")

    # Plot Elbow Curve and Silhouette Score
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Ax1: Elbow Method
    ax1.plot(list(k_range), inertias, marker="o", color="#1f77b4", linewidth=2.5, markersize=8)
    ax1.axvline(x=3, color="#d62728", linestyle="--", alpha=0.8, label="Elbow inflection point (k=3)")
    ax1.set_title("Elbow Method for Optimal k (Inertia / WCSS)")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax1.set_xticks(list(k_range))
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Ax2: Silhouette Scores
    ax2.plot(range(2, 11), silhouette_scores, marker="s", color="#2ca02c", linewidth=2.5, markersize=8)
    ax2.axvline(x=3, color="#d62728", linestyle="--", alpha=0.8, label="Domain target (k=3)")
    ax2.set_title("Silhouette Scores across k")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Mean Silhouette Coefficient")
    ax2.set_xticks(range(2, 11))
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "1_elbow_and_silhouette.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Elbow & Silhouette plot -> {output_path}")

    return inertias, silhouette_scores


def apply_pca(X_scaled, feature_names):
    """
    Applies PCA to reduce dimensionality from 4D to 2D and 3D.
    Analyzes explained variance and creates a Scree Plot.
    """
    print("\n" + "=" * 70)
    print("STEP 3: DIMENSIONALITY REDUCTION VIA PCA")
    print("=" * 70)

    pca_full = PCA(n_components=4, random_state=42)
    pca_full.fit(X_scaled)

    var_ratio = pca_full.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    print("Explained Variance Ratio per Principal Component:")
    for idx, (var, cum) in enumerate(zip(var_ratio, cum_var_ratio), start=1):
        print(f"  PC{idx}: {var*100:6.2f}% (Cumulative: {cum*100:6.2f}%)")

    # Fit 2D and 3D PCA
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    pca_3d = PCA(n_components=3, random_state=42)
    X_pca_3d = pca_3d.fit_transform(X_scaled)

    # Component loadings (how original features map to PC1 & PC2)
    loadings = pd.DataFrame(
        pca_2d.components_.T,
        columns=["PC1", "PC2"],
        index=feature_names,
    )
    print("\nPCA Loadings Matrix (Eigenvector weights):")
    print(loadings.round(3).to_string())

    # Scree Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    components = [f"PC{i}" for i in range(1, 5)]
    bars = ax.bar(components, var_ratio * 100, color="#4c72b0", alpha=0.85, label="Individual Variance (%)")
    ax.plot(components, cum_var_ratio * 100, color="#c44e52", marker="o", linewidth=2.5, label="Cumulative Variance (%)")

    for bar, var in zip(bars, var_ratio):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{var*100:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylim(0, 110)
    ax.set_title("PCA Scree Plot: Explained Variance by Component")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    scree_path = os.path.join(OUTPUT_DIR, "2_pca_scree_plot.png")
    plt.savefig(scree_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Scree plot -> {scree_path}")

    return pca_2d, X_pca_2d, pca_3d, X_pca_3d


def run_kmeans_clustering(X_scaled, X_pca_2d, k=3):
    """
    Fits K-Means with k=3 on the 4D standardized feature space.
    Calculates cluster centers in both 4D feature space and 2D PCA space.
    """
    print("\n" + "=" * 70)
    print(f"STEP 4: K-MEANS CLUSTERING (k={k})")
    print("=" * 70)

    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=300, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Compute centroids in 2D PCA space
    pca_centroids = np.zeros((k, 2))
    for cluster_id in range(k):
        pca_centroids[cluster_id] = X_pca_2d[cluster_labels == cluster_id].mean(axis=0)

    print(f"Model converged in {kmeans.n_iter_} iterations.")
    print(f"Final Inertia (WCSS): {kmeans.inertia_:.2f}")

    return kmeans, cluster_labels, pca_centroids


def map_clusters_to_labels(y_true, cluster_labels, target_names):
    """
    Uses the Hungarian Algorithm (linear_sum_assignment) on the contingency matrix
    to optimally match unsupervised cluster IDs to ground truth class labels.
    """
    print("\n" + "=" * 70)
    print("STEP 5: CLUSTER-TO-LABEL MAPPING & COMPARISON")
    print("=" * 70)

    # Construct contingency matrix
    contingency = np.zeros((len(target_names), len(target_names)), dtype=int)
    for true, pred in zip(y_true, cluster_labels):
        contingency[true, pred] += 1

    # Hungarian algorithm solves minimum weight matching, so we minimize negative overlap
    row_ind, col_ind = linear_sum_assignment(-contingency)

    # Mapping: cluster_id -> true_label_index
    cluster_to_target = {col: row for row, col in zip(row_ind, col_ind)}
    mapped_predictions = np.array([cluster_to_target[c] for c in cluster_labels])
    mapped_species_names = [target_names[cluster_to_target[c]] for c in cluster_labels]

    print("Optimal Cluster Assignment Mapping:")
    for cluster_id, target_idx in cluster_to_target.items():
        print(f"  Cluster {cluster_id}  -->  Class {target_idx} ({target_names[target_idx]})")

    return cluster_to_target, mapped_predictions, mapped_species_names, contingency


def compute_metrics(X_scaled, y_true, cluster_labels, mapped_predictions, target_names, kmeans):
    """Computes all unsupervised and supervised clustering metrics."""
    # Unsupervised Metrics
    sil_score = silhouette_score(X_scaled, cluster_labels)
    db_score = davies_bouldin_score(X_scaled, cluster_labels)
    ch_score = calinski_harabasz_score(X_scaled, cluster_labels)

    # Supervised Ground Truth Comparison Metrics
    ari = adjusted_rand_score(y_true, cluster_labels)
    ami = adjusted_mutual_info_score(y_true, cluster_labels)
    homo = homogeneity_score(y_true, cluster_labels)
    comp = completeness_score(y_true, cluster_labels)
    v_meas = v_measure_score(y_true, cluster_labels)
    accuracy = np.mean(mapped_predictions == y_true)

    print("\nClustering Performance Metrics Summary:")
    print("-" * 50)
    print(f"  Inertia (WCSS):                 {kmeans.inertia_:.2f}")
    print(f"  Silhouette Score:               {sil_score:.4f}  (Range: [-1, 1], higher is better)")
    print(f"  Davies-Bouldin Index:           {db_score:.4f}  (Lower is better)")
    print(f"  Calinski-Harabasz Index:        {ch_score:.2f}  (Higher is better)")
    print("-" * 50)
    print(f"  Clustering Accuracy (Aligned):  {accuracy * 100:.2f}%")
    print(f"  Adjusted Rand Index (ARI):      {ari:.4f}  (Similarity to true labels)")
    print(f"  Adjusted Mutual Info (AMI):     {ami:.4f}")
    print(f"  Homogeneity:                    {homo:.4f}")
    print(f"  Completeness:                   {comp:.4f}")
    print(f"  V-Measure:                      {v_meas:.4f}")
    print("-" * 50)

    print("\nDetailed Classification Report (Predicted vs True):")
    print(classification_report(y_true, mapped_predictions, target_names=target_names))

    metrics_dict = {
        "inertia": round(float(kmeans.inertia_), 2),
        "silhouette_score": round(float(sil_score), 4),
        "davies_bouldin_index": round(float(db_score), 4),
        "calinski_harabasz_index": round(float(ch_score), 2),
        "accuracy": round(float(accuracy), 4),
        "adjusted_rand_index": round(float(ari), 4),
        "adjusted_mutual_info": round(float(ami), 4),
        "homogeneity": round(float(homo), 4),
        "completeness": round(float(comp), 4),
        "v_measure": round(float(v_meas), 4),
    }

    metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"[Saved] Metrics summary JSON -> {metrics_path}")

    return metrics_dict


def generate_visualizations(df, X_pca_2d, X_pca_3d, pca_2d, pca_centroids, y_true, cluster_labels, mapped_predictions, target_names):
    """
    Generates all required and supplementary visual charts:
      - PCA 2D: Predicted Clusters vs True Species
      - Feature space (Petal Length vs Petal Width) comparison
      - Confusion / Contingency Matrix Heatmap
      - 3D PCA Scatter Plot
    """
    print("\n" + "=" * 70)
    print("STEP 6: GENERATING HIGH-RESOLUTION VISUALIZATIONS")
    print("=" * 70)

    palette_species = {"setosa": "#1f77b4", "versicolor": "#ff7f0e", "virginica": "#2ca02c"}
    colors_cluster = ["#d62728", "#9467bd", "#17becf"]

    # -------------------------------------------------------------
    # 1. 2D PCA PROJECTION: PREDICTED CLUSTERS VS TRUE LABELS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # Plot True Labels
    for idx, name in enumerate(target_names):
        mask = y_true == idx
        axes[0].scatter(
            X_pca_2d[mask, 0],
            X_pca_2d[mask, 1],
            label=f"{name.capitalize()} (n={np.sum(mask)})",
            color=palette_species[name],
            alpha=0.85,
            edgecolors="k",
            s=65,
        )
    axes[0].set_title("Ground Truth: Iris Species in PCA Space")
    axes[0].set_xlabel(f"Principal Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
    axes[0].set_ylabel(f"Principal Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Plot Predicted Clusters with Centroids
    for cluster_id in range(3):
        mask = cluster_labels == cluster_id
        axes[1].scatter(
            X_pca_2d[mask, 0],
            X_pca_2d[mask, 1],
            label=f"Cluster {cluster_id} (n={np.sum(mask)})",
            color=colors_cluster[cluster_id],
            alpha=0.85,
            edgecolors="k",
            s=65,
        )

    # Overlay Centroids
    axes[1].scatter(
        pca_centroids[:, 0],
        pca_centroids[:, 1],
        s=250,
        c="gold",
        marker="X",
        edgecolors="black",
        linewidths=1.8,
        label="Cluster Centroids",
        zorder=10,
    )
    for c_id, (cx, cy) in enumerate(pca_centroids):
        axes[1].annotate(
            f" C{c_id}",
            (cx, cy),
            fontsize=11,
            fontweight="bold",
            color="black",
            textcoords="offset points",
            xytext=(6, 6),
        )

    axes[1].set_title("Unsupervised: K-Means Predicted Clusters & Centroids")
    axes[1].set_xlabel(f"Principal Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
    axes[1].legend(loc="upper right", frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("PCA 2D Projection: Ground Truth vs K-Means (k=3)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    pca_comp_path = os.path.join(OUTPUT_DIR, "3_pca_clusters_vs_true.png")
    plt.savefig(pca_comp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] PCA Comparison Plot -> {pca_comp_path}")

    # -------------------------------------------------------------
    # 2. FEATURE SPACE SCATTER PLOT (PETAL LENGTH VS PETAL WIDTH)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # True labels in feature space
    for idx, name in enumerate(target_names):
        mask = y_true == idx
        axes[0].scatter(
            df.loc[mask, "petal length (cm)"],
            df.loc[mask, "petal width (cm)"],
            label=name.capitalize(),
            color=palette_species[name],
            alpha=0.85,
            edgecolors="k",
            s=65,
        )
    axes[0].set_title("Ground Truth: Petal Dimensions")
    axes[0].set_xlabel("Petal Length (cm)")
    axes[0].set_ylabel("Petal Width (cm)")
    axes[0].legend(loc="upper left", frameon=True)
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Predicted clusters in feature space
    for cluster_id in range(3):
        mask = cluster_labels == cluster_id
        axes[1].scatter(
            df.loc[mask, "petal length (cm)"],
            df.loc[mask, "petal width (cm)"],
            label=f"Cluster {cluster_id}",
            color=colors_cluster[cluster_id],
            alpha=0.85,
            edgecolors="k",
            s=65,
        )
    axes[1].set_title("K-Means Clusters: Petal Dimensions")
    axes[1].set_xlabel("Petal Length (cm)")
    axes[1].legend(loc="upper left", frameon=True)
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Feature Space Comparison: Petal Length vs Petal Width", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    feat_comp_path = os.path.join(OUTPUT_DIR, "4_feature_space_comparison.png")
    plt.savefig(feat_comp_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Feature Space Comparison -> {feat_comp_path}")

    # -------------------------------------------------------------
    # 3. CONFUSION MATRIX HEATMAP (MAPPED PREDICTIONS VS TRUE LABELS)
    # -------------------------------------------------------------
    cm = confusion_matrix(y_true, mapped_predictions)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=[t.capitalize() for t in target_names],
        yticklabels=[t.capitalize() for t in target_names],
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )
    ax.set_title("Confusion Matrix: Aligned K-Means vs Ground Truth", pad=15)
    ax.set_xlabel("Predicted Species (Aligned Clusters)", labelpad=10)
    ax.set_ylabel("True Iris Species", labelpad=10)

    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "5_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] Confusion Matrix Heatmap -> {cm_path}")

    # -------------------------------------------------------------
    # 4. 3D PCA SCATTER PLOT
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for cluster_id in range(3):
        mask = cluster_labels == cluster_id
        ax.scatter(
            X_pca_3d[mask, 0],
            X_pca_3d[mask, 1],
            X_pca_3d[mask, 2],
            label=f"Cluster {cluster_id}",
            color=colors_cluster[cluster_id],
            s=55,
            edgecolors="k",
            alpha=0.8,
        )
    ax.set_title("3D PCA Cluster Projection (PC1 vs PC2 vs PC3)", pad=20)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(loc="upper left")
    plt.tight_layout()
    pca3d_path = os.path.join(OUTPUT_DIR, "6_3d_pca_visualization.png")
    plt.savefig(pca3d_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] 3D PCA Plot -> {pca3d_path}")


def main():
    # Step 1: Load & Explore
    iris, df = load_and_explore_data()
    X = df[iris.feature_names].values
    y_true = df["target"].values

    # Step 2: Standardize
    scaler, X_scaled = scale_features(X)

    # Step 3: Elbow & Silhouette Analysis
    evaluate_optimal_k(X_scaled)

    # Step 4: Apply PCA
    pca_2d, X_pca_2d, pca_3d, X_pca_3d = apply_pca(X_scaled, iris.feature_names)

    # Step 5: K-Means (k=3)
    kmeans, cluster_labels, pca_centroids = run_kmeans_clustering(X_scaled, X_pca_2d, k=3)

    # Step 6: Mapping & Evaluation
    cluster_to_target, mapped_predictions, mapped_species, contingency = map_clusters_to_labels(
        y_true, cluster_labels, iris.target_names
    )
    compute_metrics(X_scaled, y_true, cluster_labels, mapped_predictions, iris.target_names, kmeans)

    # Step 7: Visualizations
    generate_visualizations(
        df,
        X_pca_2d,
        X_pca_3d,
        pca_2d,
        pca_centroids,
        y_true,
        cluster_labels,
        mapped_predictions,
        iris.target_names,
    )

    # Step 8: Save enriched dataset
    df["PC1"] = X_pca_2d[:, 0]
    df["PC2"] = X_pca_2d[:, 1]
    df["kmeans_cluster"] = cluster_labels
    df["predicted_species"] = mapped_species
    df["is_correct"] = df["species"] == df["predicted_species"]

    csv_path = os.path.join(OUTPUT_DIR, "iris_clustered_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Saved] Enriched CSV results -> {csv_path}")

    print("\n" + "=" * 70)
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
