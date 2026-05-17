# Data Imbalance

**Source**: https://aman.ai/primers/ai/data-imbalance
**Category**: Data Foundations
**Word Count**: 2438

---

Overview
Data-based Methods
Oversampling Techniques
Undersampling Techniques
Hybrid Approaches
Stratified Splitting
Loss Function-based Methods
Focal Loss
Weighted Loss Functions
Model-based Methods
Benefits of Ensemble Methods for Class Imbalance
Bagging
Boosting
Metrics-based Methods
Evaluation Metrics
Additional Diagnostic Tools (Confusion Matrix and Correlation Coefficients)
Calibration Metrics
Practical Recommendations
FAQs
How do ensemble methods help with class imbalance?
Bagging Methods (e.g., Random Forest)
Boosting Methods (e.g., AdaBoost, Gradient Boosting, XGBoost)
Ensemble of Resampled Datasets
Cost-Sensitive Learning with Ensembles
Hybrid Approaches
Key Advantages of Using Ensembles for Class Imbalance
Citation
Overview
Class imbalance arises when one or more classes in a dataset are significantly underrepresented compared to others, often leading to biased predictions by machine learning models. Models trained on imbalanced datasets may perform well for the majority class but fail to adequately capture patterns for minority classes. Addressing class imbalance is essential to ensure fair, accurate, and generalized predictions.
Below, we explore detailed techniques to handle class imbalance at the data, model, and metrics levels.
Data-based Methods
Data-based methods aim to modify the dataset to balance the class distribution before training. By transforming the data, these approaches directly impact the availability and representativeness of minority classes in the training process. Common techniques include oversampling, undersampling, hybrid methods, and data transformation.
Oversampling Techniques
Oversampling involves increasing the representation of minority classes in the dataset by duplicating or generating synthetic samples.
Random Oversampling
:
Randomly duplicates existing samples from the minority class until the class sizes are balanced.
While simple to implement, random oversampling can lead to overfitting, as it replicates the same samples multiple times without introducing new information.
Advanced Variants
:
Augmentation-Based Oversampling
:
Generates variability in minority class data by applying transformations like rotation, cropping, flipping, noise addition, or color jittering.
Particularly effective in image and text data, where augmentation introduces realistic variations without requiring additional data collection.
SMOTE Variants (Synthetic Minority Oversampling Technique)
:
SMOTE generates synthetic data points by interpolating between existing minority class samples and their nearest neighbors. Advanced SMOTE variants include:
K-Means SMOTE
:
Applies k-means clustering to the data and generates synthetic samples from minority clusters, ensuring the synthetic data aligns better with the data’s natural structure.
SMOTE-Tomek
:
Combines SMOTE with Tomek links (pairs of nearest-neighbor samples from different classes) to oversample the minority class while removing borderline and noisy samples.
SVM-SMOTE
:
Focuses on generating synthetic samples near the support vectors of a Support Vector Machine (SVM) classifier, emphasizing the critical decision boundaries.
GAN-Based Data Augmentation/Oversampling
:
Conditional GANs (cGANs)
:
Uses Generative Adversarial Networks (GANs) conditioned on class labels to create realistic and diverse synthetic samples for the minority class.
Particularly useful for high-dimensional, complex datasets like images, time-series, or text, where traditional oversampling might fail to capture nuanced patterns.
CycleGANs
for domain-specific augmentation (e.g., converting aerial images to street views).
Variational Autoencoders (VAEs)
to generate synthetic tabular data.
Undersampling Techniques
Undersampling reduces the size of the majority class to achieve a balanced dataset. This approach is effective when the majority class has redundant or non-informative samples.
Edited Nearest Neighbors (ENN)
:
Removes samples from the majority class that are misclassified by their k-nearest neighbors, ensuring that noisy and overlapping samples are eliminated, leading to cleaner decision boundaries.
Tomek Links
:
Identifies pairs of samples from different classes that are each other’s nearest neighbors and removes the majority class samples from these pairs.
Improves class separability by reducing overlap between classes.
Cluster-Based Undersampling
:
Uses clustering algorithms (e.g., k-means) to identify representative samples from the majority class, reducing redundancy while retaining critical patterns.
Preserves the diversity of majority class data, preventing loss of important information.
Hybrid Approaches
Hybrid approaches combine oversampling and undersampling techniques to leverage the benefits of both.
SMOTE + ENN
:
Applies SMOTE to oversample the minority class and ENN to remove noisy or overlapping samples from the majority class.
Results in a balanced and clean dataset, especially for datasets with significant noise or class overlap.
ADASYN + Cluster Centroids
:
ADASYN (Adaptive Synthetic Sampling) generates synthetic samples for harder-to-classify minority samples, while cluster centroids reduce the size of the majority class by representing clusters as single samples.
Ensures balanced yet simplified data for training.
Stratified Splitting
Stratified splitting ensures that the class distribution in the training, validation, and test splits matches the original dataset. This prevents data leakage and ensures that the model’s performance metrics are evaluated fairly across all classes.
Implementation:
Tools like Python’s
scikit-learn
provide a
stratify
parameter in the
train_test_split
function to maintain class proportions across splits.
Loss Function-based Methods
These methods modify the loss function to penalize misclassification of minority classes more heavily, improving model sensitivity to underrepresented data.
Focal Loss
Tailored for extreme class imbalance, Focal Loss emphasizes harder-to-classify samples by down-weighting easy samples.
\(L = -\alpha (1 - p_t)^\gamma \log(p_t).\)
Parameters:
\(\alpha\): Balances class contributions to the loss.
\(\gamma\): Focuses training on hard samples, making the model more sensitive to the minority class.
Weighted Loss Functions
Assigns higher weights to minority classes, increasing their influence on the loss and model updates.
Example:
\(w_c = \frac{N}{n_c},\)
where \(N\) is the total number of samples and \(n_c\) is the number of samples in class \(c\).
Widely supported in frameworks like TensorFlow, PyTorch, and scikit-learn.
Model-based Methods
Model-based methods adapt algorithms to emphasize minority classes, often through paradigms like ensemble approaches.
Ensemble methods combine predictions from multiple models to improve overall performance. They are particularly effective for handling class imbalance, as they can adjust focus on minority classes during training through techniques like sampling and weighting.
Benefits of Ensemble Methods for Class Imbalance
Improved Generalization
:
By combining multiple models, ensemble methods reduce the risk of bias toward the majority class and ensure robust performance across classes.
Flexible Sampling
:
Bagging and boosting can be combined with oversampling, undersampling, or hybrid sampling strategies, enhancing their adaptability to class imbalance.
Customizable Weighting
:
Most ensemble frameworks, particularly boosting methods, allow for class-based weighting in their objectives, enabling precise control over class contributions to the final model.
Enhanced Decision Boundaries
:
Ensembles, especially boosting methods, refine decision boundaries iteratively, ensuring minority class regions are not overlooked.
In practice, the choice between bagging and boosting depends on the dataset and model goals:
Bagging
is better for reducing overfitting and leveraging parallelism.
Boosting
excels in capturing complex patterns, particularly for skewed distributions, with the added advantage of class weighting options in frameworks like XGBoost and LightGBM.
Bagging
Bagging (Bootstrap Aggregating) reduces variance and improves model robustness by training multiple models on different subsets of data sampled with replacement. In the context of class imbalance, bagging methods help by:
Boosting Minority Representation
:
Resampling techniques, such as oversampling the minority class or undersampling the majority class, can be applied within each bootstrap sample.
Ensures that minority classes are adequately represented in the training data for each model.
Random Forest
:
As a bagging method, Random Forest can handle class imbalance by:
Adjusting the class distribution in each bootstrap sample.
Assigning class weights inversely proportional to their frequencies during tree construction.
Class-Specific Aggregation
:
Combines predictions across multiple models, often weighting minority class predictions higher to correct for imbalance.
Boosting
Boosting sequentially trains models, focusing on samples that previous models misclassified. It is inherently suited to handling class imbalance due to its iterative adjustment of sample weights.
Focusing on Hard-to-Classify Samples
:
Boosting algorithms (e.g., AdaBoost, Gradient Boosting) assign higher weights to misclassified samples, often aligning with minority class instances in imbalanced datasets.
Specialized Boosting Variants
:
SMOTEBoost
:
Integrates SMOTE oversampling with boosting. At each iteration, synthetic samples are generated for the minority class, ensuring better representation.
RUSBoost
:
Combines Random Undersampling (RUS) with boosting to reduce majority class dominance while maintaining minority class focus.
Class Weight Support
:
Many modern boosting frameworks, such as
XGBoost
,
LightGBM
, and
CatBoost
, allow specifying
class weights
directly in their loss functions.
Class weights allow the algorithm to penalize misclassifications of minority class samples more heavily, improving balance. Put simply, this feature prioritizes the minority class by increasing its contribution to the optimization objective, further mitigating imbalance effects.
Metrics-based Methods
Metrics play a crucial role in evaluating machine learning models, particularly for imbalanced datasets, where standard accuracy measures can be misleading. By adopting specialized metrics, practitioners can ensure that the performance evaluation aligns with the goals of addressing class imbalance and prioritizing minority class outcomes.
Evaluation Metrics
Precision, Recall, and F1-Score
:
These metrics go beyond overall accuracy by focusing on specific aspects of model performance, particularly for the minority class:
Precision
:
Represents the proportion of correctly identified positive samples out of all samples predicted as positive.
Useful in scenarios where false positives are costly, such as fraud detection.
\(\text{Precision} = \frac{TP}{TP + FP}.\)
Recall
:
Measures the proportion of actual positives correctly identified by the model.
Crucial for applications like medical diagnosis, where missing positive cases can have severe consequences.
\(\text{Recall} = \frac{TP}{TP + FN}.\)
F1-Score
:
The harmonic mean of Precision and Recall, balancing their trade-offs.
Provides a single, interpretable metric to assess a model’s focus on minority classes.
\(F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}.\)
AUC-PR (Precision-Recall Curve)
:
Evaluates performance across different thresholds by plotting Precision against Recall.
More sensitive to the minority class than ROC-AUC because it avoids the dilution caused by the dominant majority class.
Class-Specific Metrics
:
Evaluate metrics like Precision, Recall, and F1 for each class separately, offering a detailed understanding of how the model performs for the minority class versus the majority class.
Additional Diagnostic Tools (Confusion Matrix and Correlation Coefficients)
Confusion Matrix Analysis
:
Provides a granular view of model performance across all prediction outcomes, including true/false positives and negatives.
Enables targeted optimization for minority classes by identifying patterns in errors.
Matthews Correlation Coefficient (MCC)
:
A comprehensive metric for binary classification that considers true and false positives and negatives.
Particularly robust for imbalanced datasets as it evaluates all four quadrants of the confusion matrix, providing a balanced measure.
\(\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.\)
Cohen’s Kappa
:
Measures agreement between predicted and actual labels, adjusted for chance.
Effective for class imbalance, as it accounts for the disparity in class proportions.
\(\text{Kappa} = \frac{P_o - P_e}{1 - P_e},\)
where (P_o) is the observed agreement and (P_e) is the agreement expected by chance.
Calibration Metrics
Brier Score
:
Assesses the accuracy of probabilistic predictions by penalizing confidence in incorrect predictions and rewarding well-calibrated probabilities.
Especially relevant for imbalanced datasets, where models often exhibit calibration issues due to overconfidence in the majority class.
\(\text{Brier Score} = \frac{1}{N} \sum_{i=1}^{N} (f_i - y_i)^2,\)
where (f_i) is the predicted probability for sample (i) and (y_i) is the actual label.
Expected Calibration Error (ECE)
:
Measures the difference between predicted probabilities and actual outcomes, providing insight into the reliability of a model’s probabilistic outputs.
Practical Recommendations
Data-Level Techniques
:
Employ
SMOTE + Tomek Links
to oversample the minority class and remove overlapping samples that introduce noise.
Algorithm Adjustments
:
Train a
Weighted XGBoost
model with
Focal Loss
to dynamically focus on difficult samples, especially in imbalanced datasets.
Evaluation
:
Prioritize metrics that emphasize the minority class, such as
Precision-Recall curves
,
F1-Scores
, and
MCC
.
Calibration
:
Validate model outputs with
Brier Score
and calibration plots to ensure reliable probabilistic predictions.
Class imbalance can be addressed effectively by leveraging a combination of these methods, tuned to the problem’s specific needs.
FAQs
How do ensemble methods help with class imbalance?
Ensemble methods are effective tools for addressing class imbalance, as they combine multiple models to improve overall performance, reduce overfitting, and mitigate the bias toward majority classes. By amplifying the signal from minority class data and leveraging the diversity of models, these methods enhance prediction accuracy and fairness across all classes. When paired with complementary techniques such as resampling, adjusting class weights, or generating synthetic data, ensemble methods can yield even more robust results in handling imbalanced datasets.
Bagging Methods (e.g., Random Forest)
How It Helps:
Bagging trains multiple models on different bootstrapped (randomly sampled with replacement) subsets of the data.
You can apply techniques like oversampling the minority class or undersampling the majority class within each bootstrapped sample to improve representation of minority classes.
Random Forests average predictions across trees, which helps mitigate the bias introduced by imbalanced data.
Advantages:
Reduces variance and prevents overfitting.
Can handle imbalance if combined with balanced sampling strategies.
Boosting Methods (e.g., AdaBoost, Gradient Boosting, XGBoost)
How It Helps:
Boosting focuses on correcting the mistakes of previous models by assigning higher weights to misclassified instances.
In the case of imbalanced datasets, boosting naturally places more emphasis on minority class samples, as they are more likely to be misclassified in early iterations.
Many boosting frameworks (e.g., XGBoost, LightGBM) allow specifying
class weights
, which further prioritize the minority class.
Advantages:
Effective at focusing on hard-to-classify samples (often minority class).
Customizable with parameters like learning rate and class weights.
Ensemble of Resampled Datasets
How It Helps:
Build multiple models, each trained on a dataset that has been resampled to balance the classes.
For example:
Over-sampling:
Duplicate samples of the minority class.
Under-sampling:
Reduce samples of the majority class.
Combine predictions using voting or averaging to reduce individual model biases.
Advantages:
Balances class representation while maintaining diversity among models.
Reduces overfitting to the majority class.
Cost-Sensitive Learning with Ensembles
How It Helps:
Modify the objective function of ensemble models to include misclassification costs.
Penalize misclassifications of the minority class more heavily, forcing the model to focus on getting those predictions right.
Many frameworks, such as XGBoost, support custom loss functions that incorporate class imbalance.
Advantages:
Directly addresses the imbalance by prioritizing the minority class.
Avoids the need for resampling.
Hybrid Approaches
How It Helps:
Combine ensemble methods with other imbalance techniques, such as SMOTE (Synthetic Minority Oversampling Technique).
For example:
Use SMOTE to generate synthetic samples for the minority class, then train a Random Forest or XGBoost model.
Advantages:
Leverages the strengths of both resampling and ensemble learning.
Can yield high performance even for severely imbalanced datasets.
Key Advantages of Using Ensembles for Class Imbalance
Improved Robustness:
Ensembles aggregate predictions, reducing the likelihood of bias from a single model.
Focus on Hard Cases:
Methods like boosting inherently focus on hard-to-classify samples, which are often from the minority class.
Flexibility:
Many ensemble methods can integrate class weights or cost-sensitive learning to handle imbalance directly.
Versatility:
Ensembles can be combined with other preprocessing or algorithmic approaches for greater effectiveness.
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DataImbalance,
  title   = {Data Imbalance},
  author  = {Chadha, Aman},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}