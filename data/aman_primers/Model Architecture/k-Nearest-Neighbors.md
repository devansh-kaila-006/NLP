# k-Nearest Neighbors

**Source**: https://aman.ai/primers/ai/k-nearest-neighbors
**Category**: Model Architecture
**Word Count**: 2596

---

k-Nearest Neighbors (kNN)
Overview
Algorithm Description
Training Phase
Prediction Phase
Distance Metrics
Parameters
Stopping Criterion
Advantages
Disadvantages
Example (Classification)
Dataset
Test Point: \((5, 5)\)
FAQ: What are some methods that do not require a predefined \(k\) value for nearest neighbor selection?
Distance-Weighted kNN
Radius-Based Neighbors
Adaptive \(k\)-Nearest Neighbors
Local Outlier Factor (LOF) or Similar Techniques
Ensemble of kNN Models
Dynamic kNN
Data-Driven Methods (Validation or Cross-Validation)
Comparison of Approaches
Comparison: k-Nearest Neighbors vs. k-Means Clustering
Key Takeaways
Citation
k-Nearest Neighbors (kNN)
Overview
Type:
Supervised learning algorithm.
Category:
Non-parametric.
kNN is classified as a non-parametric model because it does not make assumptions about the underlying data distribution, making it flexible for a wide range of applications.
Primary Use Case:
Classification:
Assigns discrete class labels to a new data point by evaluating the labels of nearby points.
Regression:
Predicts continuous numerical values by considering the values of neighboring points.
Key Idea:
The fundamental principle of kNN revolves around the notion that similar data points are located close to each other in the feature space. Thus, the label or value of a new data point can be inferred based on the labels or values of its closest neighbors.
Algorithm Description
Training Phase
kNN is a
lazy learning algorithm
, meaning it does not construct a specific model during training. Instead:
The algorithm simply memorizes the dataset, denoted as:
\(\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n\)
Here:
\(x_i\) is the feature vector representing the \(i^{th}\) data point.
\(y_i\) is the corresponding label or output (a class label for classification or a numerical value for regression).
As there is no explicit model construction, all computation is deferred to the prediction phase.
Prediction Phase
For a given test point \(x_t\):
Compute Distances:
Measure the distance between the test point \(x_t\) and each training data point \(x_i\) in the dataset using a selected distance metric \(d(x_t, x_i)\).
Common distance metrics include
Euclidean distance
,
Manhattan distance
, and others (discussed later).
Sort Neighbors:
Rank the training points based on their computed distances to \(x_t\) in ascending order.
Select Nearest Neighbors:
Identify the \(k\)-nearest neighbors (i.e., the \(k\) training points with the smallest distances).
Make Predictions:
For
classification tasks
:
Assign the class label that appears most frequently among the \(k\) neighbors. This is expressed mathematically as:
\(\hat{y}_t = \text{mode}\{y_i \mid x_i \in \text{neighbors of } x_t\}\)
For
regression tasks
:
Compute the prediction as the mean or weighted mean of the output values of the \(k\) neighbors:
\(\hat{y}_t = \frac{1}{k} \sum_{x_i \in \text{neighbors of } x_t} y_i\)
Distance Metrics
The choice of distance metric has a profound impact on the accuracy and efficiency of kNN. Popular metrics include:
Euclidean Distance:
The most commonly used metric for continuous numerical data. It calculates the straight-line distance between two points in \(n\)-dimensional space:
\(d(\mathbf{x}, \mathbf{y}) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}\)
Suitable when all features are on similar scales.
Manhattan Distance (L1 Norm):
Measures the distance by summing the absolute differences along each dimension:
\(d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^n |x_i - y_i|\)
Often used when capturing the magnitude of differences is more meaningful than the straight-line distance.
Minkowski Distance:
A generalization of Euclidean and Manhattan distances, controlled by a parameter \(p\):
\(d(\mathbf{x}, \mathbf{y}) = \left(\sum_{i=1}^n |x_i - y_i|^p\right)^{\frac{1}{p}}\)
Special cases:
\(p = 2\): Euclidean distance.
\(p = 1\): Manhattan distance.
Hamming Distance:
Used for categorical features, it counts the number of differing feature values:
\(d(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^n \mathbb{1}(x_i \neq y_i)\)
where \(\mathbb{1}(\cdot)\) is an indicator function that outputs 1 if the condition is true, otherwise 0.
Parameters
Number of Neighbors (\(k\)):
\(k\) specifies how many neighbors influence the prediction.
Small \(k\):
Sensitive to noise, as predictions depend on fewer data points.
Risks overfitting the data.
Large \(k\):
Incorporates more neighbors, leading to smoother predictions.
Risks underfitting, as it may dilute the influence of closer neighbors.
Optimal \(k\):
Determined using techniques such as cross-validation, which evaluates performance on a validation set for various \(k\) values.
Distance Metric:
The metric must suit the data type. For instance:
Use
Euclidean distance
for numerical data.
Use
Hamming distance
for categorical data.
Weighted kNN:
In some cases, assigning higher importance to closer neighbors improves predictions. A common weighting scheme is:
\(w_i = \frac{1}{d(x_t, x_i)^2}\)
Stopping Criterion
kNN does not involve iterative updates. Predictions conclude once:
Distances are calculated.
\(k\) neighbors are identified.
The output (classification or regression) is determined.
Advantages
Intuitive and Easy to Implement:
kNN is straightforward and does not require extensive mathematical formulations or parameter tuning.
Adaptable to Various Data Types:
Works with both numerical and categorical data.
No Training Phase:
Reduces computational costs during training since no explicit model is built.
Disadvantages
Computationally Expensive During Prediction:
Requires distance computation for every training data point at prediction time, which can be slow for large datasets.
Sensitive to Feature Scaling:
Features on different scales can dominate the distance metric unless standardized.
Performance Relies on Parameter Choices:
Poor choices for \(k\) or distance metrics can significantly degrade model performance.
Sensitive to Irrelevant Features:
Irrelevant or noisy features can skew the distance calculations, reducing accuracy.
Example (Classification)
Dataset
Point
Feature 1 (x₁)
Feature 2 (x₂)
Label
A
2
4
Red
B
4
4
Blue
C
4
6
Blue
D
6
4
Red
Test Point: \((5, 5)\)
Compute distances from \((5, 5)\) to all training points.
Sort distances and pick the \(k\)-nearest points.
Assign the majority class as the label for \((5, 5)\).
FAQ: What are some methods that do not require a predefined \(k\) value for nearest neighbor selection?
There are several approaches to adapt the \(k\)-Nearest Neighbors (kNN) algorithm to avoid requiring a predefined \(k\) value. These methods focus on dynamically determining the value of \(k\) or using alternative mechanisms to achieve robust and adaptive nearest neighbor selection. Below are some common techniques explained in greater detail:
Distance-Weighted kNN
Concept
: Instead of relying on a fixed \(k\), the algorithm assigns weights to each neighbor based on its distance to the query point. This approach emphasizes the contribution of closer neighbors while reducing the influence of more distant ones.
Implementation
:
Use weighting functions such as:
Inverse Distance Weighting
: \(w_i = \frac{1}{d_i}\), where \(d_i\) is the distance to the \(i\)-th neighbor.
Gaussian Function
: \(w_i = e^{-\frac{d_i^2}{\sigma^2}}\), where \(\sigma\) controls the sensitivity to distance.
Advantages
:
Effectively adapts to local density variations without requiring a fixed neighbor count.
Mitigates issues arising from abrupt cutoff at a predefined \(k\).
Applications
: Commonly used in regression tasks and probabilistic classification.
Radius-Based Neighbors
Concept
: Select neighbors within a specified radius \(r\) around the query point instead of a fixed \(k\) number of neighbors.
Implementation
:
Define a radius \(r\) and include all data points within this radius as neighbors.
Optionally use an
adaptive radius
that depends on the local data density or a scaling factor.
Advantages
:
Eliminates the need for a fixed \(k\), adapting to local variations in data distribution.
Suitable for highly imbalanced datasets or datasets with varying density clusters.
Challenges
:
Choosing an appropriate radius \(r\) can be complex, especially for high-dimensional data.
Extensions
:
Dynamically adjust \(r\) based on density estimation techniques or cross-validation.
Adaptive \(k\)-Nearest Neighbors
Concept
: Dynamically adjust \(k\) based on local data characteristics, such as density or classification stability.
Implementation
:
Increase \(k\) incrementally until:
Classification or regression results stabilize.
A confidence threshold (e.g., prediction probability) is met.
Use cross-validation to determine \(k\) for each query point or region of the dataset.
Advantages
:
Allows for tailored neighborhood selection, improving accuracy in heterogeneous datasets.
Works well in scenarios with overlapping class boundaries or noisy data.
Applications
:
Often used in domains with non-uniform data distributions or high variability.
Local Outlier Factor (LOF) or Similar Techniques
Concept
: Techniques like LOF generalize neighborhood selection by computing local densities rather than relying on a strict count of neighbors.
Implementation
:
Compare the density of the query point’s neighborhood to the densities of neighboring points.
Use relative density to determine “neighborhood influence” dynamically.
Advantages
:
Focuses on meaningful neighbors by accounting for density-based variation.
Robust against outliers and sparse regions in the dataset.
Applications
:
Anomaly detection, density-based clustering, and non-parametric density estimation.
Ensemble of kNN Models
Concept
: Combine predictions from multiple kNN models built with varying \(k\) values to avoid reliance on a specific \(k\).
Implementation
:
Train several kNN models with different \(k\) values.
Aggregate predictions using:
Majority Voting
for classification tasks.
Weighted Averaging
for regression tasks.
Advantages
:
Reduces sensitivity to the choice of \(k\).
Improves robustness and generalization performance.
Challenges
:
Increased computational complexity due to multiple models.
Applications
:
Used in ensemble learning frameworks to boost predictive accuracy.
Dynamic kNN
Concept
: Adjust \(k\) dynamically for each query point based on specific criteria or thresholds.
Implementation
:
Stop increasing \(k\) when:
The cumulative distance of the \(k\)-nearest neighbors exceeds a predefined threshold.
The classification result achieves a minimum level of confidence.
Use statistical measures (e.g., confidence intervals or error bounds) to guide \(k\) selection.
Advantages
:
Provides flexibility and adapts to query-specific requirements.
Ensures that neighborhood selection aligns with the task’s performance criteria.
Applications
:
Real-time decision-making systems, active learning, and streaming data analysis.
Data-Driven Methods (Validation or Cross-Validation)
Concept
: Automatically select \(k\) through data-driven optimization techniques, eliminating the need for user input.
Implementation
:
Use
cross-validation
to evaluate different \(k\) values on a validation set and select the optimal one.
Apply advanced optimization techniques, such as:
Grid Search
: Explore a predefined range of \(k\) values systematically.
Bayesian Optimization
: Probabilistically search for the best \(k\) based on performance metrics.
Advantages
:
Provides a principled approach to \(k\) selection.
Optimizes the balance between bias and variance for the specific dataset.
Challenges
:
Computational overhead due to repeated evaluations.
Applications
:
Suitable for model tuning in supervised learning tasks.
Comparison of Approaches
Method
Concept
Implementation
Advantages
Challenges
Applications
Distance-Weighted kNN
Weights neighbors based on distance to query point instead of fixed \(k\).
- Use weighting functions like inverse distance weighting or Gaussian functions.
- Weight reduces influence of distant points.
- Adapts to local density variations.
- Avoids abrupt cutoff issues.
- Improves robustness in regression and probabilistic tasks.
- Requires choice of weighting function and parameters (e.g., \(\sigma\)).
- Computationally expensive for large datasets.
- Regression tasks.
- Probabilistic classification.
Radius-Based Neighbors
Selects neighbors within a specified radius \(r\) instead of a fixed count.
- Define a radius \(r\) around query point.
- Optionally, use adaptive radius based on data density.
- Adapts to varying data densities.
- Suitable for imbalanced datasets.
- Choosing \(r\) is complex, especially in high-dimensional spaces.
- Performance depends on radius selection mechanism.
- Highly imbalanced datasets.
- Datasets with varying density clusters.
Adaptive $$k$$-NN
Dynamically adjusts \(k\) based on local data characteristics.
- Incrementally increase \(k\) until stability or a confidence threshold is achieved.
- Use cross-validation to tailor \(k\) per query or region.
- Improves accuracy in heterogeneous datasets.
- Handles overlapping class boundaries.
- Robust to noisy data.
- Computationally intensive.
- May require domain-specific thresholds for stability or confidence.
- Non-uniform data distributions.
- Noisy or complex datasets.
Local Outlier Factor
Selects neighbors dynamically based on relative density rather than a strict count.
- Compare local density of query point with densities of neighbors.
- Use relative density to determine influence dynamically.
- Adapts to density variations.
- Robust to outliers and sparse regions.
- Parameter sensitivity (e.g., scale of density).
- High-dimensional data can complicate density estimation.
- Anomaly detection.
- Density-based clustering.
- Non-parametric density estimation.
Ensemble of kNN Models
Combines predictions from multiple kNN models with varying \(k\) values.
- Train several kNN models using different \(k\) values.
- Aggregate predictions via majority voting (classification) or weighted averaging (regression).
- Reduces sensitivity to $$k$$ choice.
- Improves robustness and generalization.
- Increased computational complexity.
- Requires careful aggregation strategy.
- Ensemble learning frameworks.
- Tasks requiring robust predictive accuracy.
Dynamic kNN
Adjusts \(k\) dynamically for each query based on criteria like confidence or distance thresholds.
- Increase \(k\) until:
- Cumulative distance of neighbors exceeds a threshold.
- Classification result achieves minimum confidence.
- Use statistical measures to guide \(k\) selection.
- Query-specific adaptability.
- Ensures alignment with task performance needs.
- Flexible for diverse scenarios.
- Requires well-defined criteria or thresholds for \(k\) selection.
- High computational demand for real-time applications.
- Real-time decision-making.
- Active learning.
- Streaming data analysis.
Data-Driven Methods
Selects \(k\) through validation or cross-validation on data, avoiding user input.
- Use cross-validation to evaluate and select \(k\).
- Employ optimization techniques like grid search or Bayesian optimization to find the best \(k\).
- Provides principled $$k$$ selection.
- Balances bias and variance for specific datasets.
- High computational overhead due to repeated evaluations.
- Optimization algorithms may require significant tuning.
- Model tuning in supervised learning.
- Optimizing $$k$$ for varied datasets and performance metrics.
Comparison: k-Nearest Neighbors vs. k-Means Clustering
Here is a detailed comparative analysis of k-Nearest Neighbors and k-Means Clustering:
Aspect
k-Nearest Neighbors (kNN)
k-Means Clustering
Type of Algorithm
Supervised learning algorithm (classification or regression).
Unsupervised learning algorithm (clustering).
Purpose
Predicts the label or value for a given data point based on its neighbors.
Groups data into \(k\) clusters based on similarity.
Input Requirements
Labeled data for training (requires both features and target values).
Unlabeled data (only features are required).
Working Mechanism
Finds the \(k\)-nearest points in the training dataset to a given query point and uses their labels to make predictions.
Iteratively partitions data into \(k\) clusters by minimizing intra-cluster variance.
Distance Metric
Typically uses Euclidean distance, but other metrics like Manhattan or Minkowski can also be used.
Uses Euclidean distance (or other metrics) to compute cluster centroids.
Output
Predicted label (classification) or value (regression).
Cluster assignments for each data point.
Training Phase
No explicit training; kNN is a lazy learner and computes distances during prediction.
Training involves multiple iterations to adjust centroids and assign clusters.
Prediction Phase
Involves computing distances from the query point to all training points.
Assigns new data points to the nearest cluster based on the trained centroids.
Scalability
Not scalable; high computational cost for large datasets due to distance calculations during prediction.
Scalable with optimizations; faster for large datasets after training.
Parameters
Number of neighbors (\(k\)) and distance metric.
Number of clusters (\(k\)) and initialization of centroids.
Sensitivity to Parameters
Sensitive to the choice of \(k\); inappropriate \(k\) can lead to overfitting or underfitting.
Sensitive to the choice of \(k\) and initialization of centroids; poor initialization can lead to suboptimal clustering.
Interpretability
Intuitive; directly uses neighboring points for prediction.
Less intuitive; requires interpreting clusters.
Handling of Outliers
Outliers can strongly influence predictions by affecting the nearest neighbors.
Outliers can distort cluster centroids and lead to poor clustering.
Applications
Classification (e.g., image recognition, fraud detection) and regression.
Clustering (e.g., customer segmentation, document classification).
Strengths
Simple, effective, and easy to implement; no need for explicit training.
Works well for clustering large datasets; discovers inherent structure.
Weaknesses
Computationally expensive during prediction; performance decreases with irrelevant features.
Sensitive to the choice of \(k\); may converge to local minima.
Key Takeaways
kNN:
A supervised algorithm used for prediction tasks, often with low computational overhead in the training phase but expensive at prediction time.
k-Means:
An unsupervised algorithm used to discover groupings in data, efficient for partitioning compact clusters but sensitive to initialization.
Both algorithms rely heavily on distance metrics, making feature scaling (e.g., normalization or standardization) critical for their performance.
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DistilledKNN,
  title   = {k-Nearest Neighbors},
  author  = {Chadha, Aman},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}