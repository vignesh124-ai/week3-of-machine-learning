# Mini Project 3: Iris Flower Clustering Project

An end-to-end Unsupervised Machine Learning and Dimensionality Reduction project implementing **K-Means Clustering ($k=3$)** and **Principal Component Analysis (PCA)** on the classic Iris dataset.

---

## 📌 Project Overview
- **Dataset**: Fisher's Iris Dataset (150 samples, 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width; 3 classes: *Setosa*, *Versicolor*, *Virginica*)
- **Core Tasks**:
  1. Feature Standardization using `StandardScaler`.
  2. Optimal Cluster Determination via **Elbow Method (WCSS)** and **Silhouette Analysis**.
  3. Dimensionality Reduction via **PCA** (capturing **95.81%** variance in 2D).
  4. Unsupervised **K-Means Clustering ($k=3$)**.
  5. Multi-dimensional Visualizations: 2D PCA, 3D PCA, Feature space (Petal dimensions), and Centroid mapping.
  6. Optimal cluster-to-label mapping using the **Hungarian Algorithm** (`linear_sum_assignment`).
  7. Rigorous Quantitative Evaluation (Confusion Matrix, Classification Report, Silhouette Score, ARI, AMI, V-Measure).

---

## 🗂️ Project Structure

```text
iris_clustering/
├── Iris_Flower_Clustering.ipynb    # Interactive Jupyter Notebook walkthrough
├── iris_clustering.py              # Production Python script
├── outputs/
│   ├── 1_elbow_and_silhouette.png  # Elbow Method & Silhouette Score curve
│   ├── 2_pca_scree_plot.png        # PCA Explained Variance Scree plot
│   ├── 3_pca_clusters_vs_true.png  # 2D PCA: True Species vs K-Means Clusters & Centroids
│   ├── 4_feature_space_comparison.png # Petal Length vs Width comparison
│   ├── 5_confusion_matrix.png      # Aligned Confusion Matrix Heatmap
│   ├── 6_3d_pca_visualization.png  # 3D PCA Scatter Plot
│   ├── evaluation_metrics.json     # Quantitative scores (ARI, AMI, Silhouette, etc.)
│   └── iris_clustered_results.csv  # Dataset enriched with PCA components & cluster predictions
└── README.md                       # Project documentation
```

---

## 🔬 Key Results & Evaluation Metrics

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Clustering Accuracy (Aligned)** | **83.33%** | 125 out of 150 points correctly matched |
| **Silhouette Score ($k=3$)** | **0.4599** | Well-separated, dense clusters |
| **Adjusted Rand Index (ARI)** | **0.6201** | Substantial agreement with ground truth |
| **Adjusted Mutual Info (AMI)** | **0.6552** | High shared mutual information |
| **Homogeneity / Completeness** | **0.659 / 0.660** | Balanced cluster composition |
| **PCA 2D Explained Variance** | **95.81%** | PC1 (72.96%) + PC2 (22.85%) |
| **Final K-Means Inertia (WCSS)** | **139.82** | Converged within 4 iterations |

### Classification Report (Aligned Unsupervised Clusters vs True Labels)
```
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        50
  versicolor       0.74      0.78      0.76        50
   virginica       0.77      0.72      0.74        50

    accuracy                           0.83       150
   macro avg       0.83      0.83      0.83       150
weighted avg       0.83      0.83      0.83       150
```

---

## 🚀 How to Run

### 1. Run via Python Command Line:
```bash
python iris_clustering.py
```
This executes the full pipeline and generates all plots inside `outputs/`.

### 2. Run via Jupyter Notebook:
Launch Jupyter Lab or Notebook:
```bash
jupyter notebook Iris_Flower_Clustering.ipynb
```

---

## 💡 Key Biological & Geometric Insights
1. **Iris Setosa is Linearly Separable**:
   - Both in 4D feature space and 2D PCA space, Setosa forms an isolated cluster. K-Means achieves **100% precision and 100% recall** on Setosa.
2. **Versicolor vs Virginica Overlap**:
   - In nature, *Iris versicolor* and *Iris virginica* share overlapping petal morphology.
   - Because K-Means assumes spherical clusters with equal variance, points near the boundary (14 Versicolor and 11 Virginica samples) cross the Voronoi partition, resulting in an expected unsupervised accuracy of 83.33%.
3. **Dimensionality Reduction with PCA**:
   - PC1 is primarily driven by Petal Length (loading 0.580) and Petal Width (0.565).
   - PC2 is dominated by Sepal Width (loading 0.923).
   - Compressing from 4 dimensions to 2 dimensions preserves almost all structural topology while enabling intuitive visualization.
