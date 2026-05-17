# Naive Bayes

**Source**: https://aman.ai/primers/ai/naive-bayes
**Category**: Model Architecture
**Word Count**: 1106

---

Overview
From Bayes’ theorem \(\rightarrow\) Naive Bayes
Example: Applying Bayes’ Theorem with All Terms
Given Data
Calculating Posterior Probabilities
Conclusion
Key Concepts
Bayesian Probability
Conditional Independence Assumption
Types of Naive Bayes Classifiers
Parameter Estimation
Loss Function
Applications
Text Classification and Sentiment Analysis
Medical Diagnosis
Email Spam Filtering
Recommendation Systems
Real-time Prediction
Pros and Cons
Pros
Cons
Example
Problem: Spam Detection
Step 1: Compute Priors
Step 2: Compute Likelihoods
Step 3: Compute Posterior
Step 4: Classification
Citation
Overview
Naive Bayes is a family of probabilistic algorithms based on Bayes’ theorem, with the “naive” assumption of conditional independence between every pair of features given the class label. It is widely used for classification tasks due to its simplicity, efficiency, and interpretability.
Naive Bayes algorithms are primarily used in supervised learning tasks for classification problems. The approach applies Bayes’ theorem to calculate the probability of a given sample belonging to each possible class and assigns the sample to the class with the highest posterior probability.
Despite its simplifying assumptions, Naive Bayes remains a robust and efficient tool for a wide range of practical applications, particularly in domains like text analysis and medical diagnostics. Its simplicity and interpretability make it a popular choice for baseline models and quick analyses.
From Bayes’ theorem \(\rightarrow\) Naive Bayes
Bayes’ theorem is expressed as:
\[P(C|X) = \frac{P(X|C)P(C)}{P(X)}\]
where:
\(P(C \mid X)\) is the posterior probability of the class \(C\) given the data \(X\).
\(P(X \mid C)\) is the likelihood of data \(X\) given class \(C\).
\(P(C)\) is the prior probability of class \(C\).
\(P(X)\) is the marginal probability of data \(X\).
The “naive” assumption simplifies the likelihood \(P(X \mid C)\) by assuming that features are conditionally independent given the class. For \(n\) features \(x_1, x_2, \dots, x_n\), this assumption allows us to express:
\[P(X|C) = P(x_1, x_2, \dots, x_n | C) = \prod_{i=1}^n P(x_i | C)\]
Example: Applying Bayes’ Theorem with All Terms
To understand the terms in Bayes’ theorem, consider a simple spam email classification problem. Suppose we want to classify an email as either “Spam” (S) or “Not Spam” (\(\neg S\)) based on a feature \(X\), which is the presence of the word “offer” in the email.
Given Data
Prior probabilities
:
\(P(S) = 0.2\) (20% of emails are spam).
\(P(\neg S) = 0.8\) (80% of emails are not spam).
Likelihoods
:
\(P(X \mid S) = 0.9\) (90% of spam emails contain the word “offer”).
\(P(X \mid \neg S) = 0.1\) (10% of non-spam emails contain the word “offer”).
Marginal probability
:
\(P(X) = P(X \mid S)P(S) + P(X \mid \neg S)P(\neg S)\):
\(P(X) = (0.9 \cdot 0.2) + (0.1 \cdot 0.8) = 0.18 + 0.08 = 0.26.\)
Calculating Posterior Probabilities
Using Bayes’ theorem:
For Spam
:
\(P(S | X) = \frac{P(X | S)P(S)}{P(X)} = \frac{0.9 \cdot 0.2}{0.26} = \frac{0.18}{0.26} \approx 0.692.\)
For Not Spam
:
\(P(\neg S | X) = \frac{P(X | \neg S)P(\neg S)}{P(X)} = \frac{0.1 \cdot 0.8}{0.26} = \frac{0.08}{0.26} \approx 0.308.\)
Conclusion
The posterior probabilities indicate that the email is more likely to be spam (\(P(S \mid X) \approx 0.692\)) than not spam (\(P(\neg S \mid X) \approx 0.308\)). Thus, the email would be classified as spam based on this model.
This example demonstrates the roles of all terms in Bayes’ theorem:
Prior probabilities
(\(P(S)\) and \(P(\neg S)\)) represent the base rates of each class.
Likelihoods
(\(P(X \mid S)\) and \(P(X \mid \neg S)\)) quantify the probability of observing the feature given each class.
Marginal probability
(\(P(X)\)) ensures proper normalization of posterior probabilities.
Posterior probability
(\(P(S \mid X)\)) provides the final classification decision.
Key Concepts
Bayesian Probability
The foundation of Naive Bayes lies in Bayesian probability, where we update our prior beliefs based on new evidence.
Conditional Independence Assumption
The algorithm assumes that features are independent given the class. This assumption simplifies computation but may not always hold in real-world data.
Types of Naive Bayes Classifiers
Depending on the type of data, different variations of Naive Bayes are used:
Gaussian Naive Bayes
: Assumes continuous data is normally distributed.
Multinomial Naive Bayes
: Suitable for discrete count data, such as word frequencies in text classification.
Bernoulli Naive Bayes
: Suitable for binary/boolean features.
Parameter Estimation
Parameters \(P(C)\) (prior) and \(P(x_i \mid C)\) (likelihood) are estimated using Maximum Likelihood Estimation (MLE), which counts the occurrences in the training data to compute probabilities.
Loss Function
The goal of Naive Bayes is to maximize the posterior probability. Alternatively, we minimize a loss function based on the log-likelihood. The negative log-likelihood (NLL) is commonly used as the loss function:
\[\text{NLL} = - \sum_{i=1}^N \log P(C_i | X_i)\]
where \(N\) is the number of training samples, \(C_i\) is the true class label for sample \(i\), and \(P(C_i  \mid  X_i)\) is the predicted probability for the correct class.
Applications
Text Classification and Sentiment Analysis
Naive Bayes is frequently used in Natural Language Processing (NLP) tasks such as spam detection, sentiment analysis, and document classification due to its efficiency with high-dimensional data.
Medical Diagnosis
Useful for predicting diseases based on symptoms, assuming conditional independence between symptoms.
Email Spam Filtering
Detects spam emails by analyzing the frequency of specific words in the email content.
Recommendation Systems
Predicts user preferences based on past behaviors.
Real-time Prediction
Due to its computational efficiency, Naive Bayes is effective in real-time applications.
Pros and Cons
Pros
Simplicity
: Easy to implement and interpret.
Efficiency
: Computationally fast, especially with large datasets.
Scalability
: Handles high-dimensional data well, such as text data.
No Iterative Parameter Estimation
: Training involves simple computations.
Cons
Conditional Independence Assumption
: Rarely holds in practice, which may degrade performance.
Limited Expressiveness
: Assumes simple relationships between features and the target variable.
Data Imbalance
: Sensitive to imbalanced datasets, as probabilities can be dominated by frequent classes.
Feature Engineering
: May require careful preprocessing and feature selection for optimal performance.
Example
Problem: Spam Detection
Suppose we have two classes: \(C_1\) (Spam) and \(C_2\) (Not Spam). The features are words in an email (\(X = \{x_1, x_2, x_3, \dots, x_n\}\)).
Step 1: Compute Priors
\[P(C_1) = \frac{\text{Number of Spam Emails}}{\text{Total Emails}}\]

\[P(C_2) = \frac{\text{Number of Not Spam Emails}}{\text{Total Emails}}\]
Step 2: Compute Likelihoods
For each word \(x_i\), compute \(P(x_i  \mid  C_1)\) and \(P(x_i  \mid  C_2)\) using word frequencies.
Step 3: Compute Posterior
For a new email, compute:
\[P(C_1 | X) \propto P(C_1) \prod_{i=1}^n P(x_i | C_1)\]

\[P(C_2 | X) \propto P(C_2) \prod_{i=1}^n P(x_i | C_2)\]
Step 4: Classification
Assign the email to the class with the highest posterior probability.
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DistilledNaiveBayes,
  title   = {Naive Bayes},
  author  = {Chadha, Aman},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}