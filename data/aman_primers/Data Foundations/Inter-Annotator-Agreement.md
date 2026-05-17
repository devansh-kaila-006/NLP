# Inter-Annotator Agreement

**Source**: https://aman.ai/primers/ai/inter-annotator-agreement
**Category**: Data Foundations
**Word Count**: 5440

---

Overview
Why Measure Inter-Annotator Agreement?
Types of Data for Agreement
Classical Metrics for Inter-Annotator Agreement
Cohen’s Kappa (\(\kappa\))
Scott’s Pi (\(\pi\))
Fleiss’ Kappa (\(\kappa\))
Krippendorff’s Alpha (\(\alpha\))
Correlation-Based Measures
Matthews Correlation Coefficient (MCC)
Comparative Analysis
Why Classical Metrics Fall Short for Distributional Annotations
Emergence of Distributional Annotations
Limitations of Classical Agreement Metrics
Correlation-Based Measures as Partial Workarounds
Pearson’s Correlation Coefficient (\(r\))
Spearman’s Rank Correlation Coefficient (\(\rho\))
Concordance Correlation Coefficient (CCC)
When Correlation-Based Measures Are Still Appropriate
Transition to Divergence-Based Measures
Distributional Agreement Metrics
Total Variation (TV) Distance
Kullback–Leibler (KL) Divergence
Jensen–Shannon (JS) Divergence
Comparative Analysis
Correlation Measures: Pearson’s \(r\), Spearman’s \(\rho\), and Concordance Correlation Coefficient (CCC)
Divergence Measures: TV, KL, and JS Divergence
Practical Considerations for Inter-Annotator Agreement
Choosing a Metric by Data Type
Interpreting Agreement Levels
Handling Annotator Bias and Class Imbalance
Missing Data and Sparse Annotations
Computational Considerations
Mathematical Relationships Between TV, KL, and JS
Pinsker’s Inequality (KL vs. TV)
Lower Bound on KL via TV
JS Divergence Related to KL and TV
Comparative Analysis of Theoretical Relationships
Putting It All Together: A Workflow for Measuring IAA
Step 1: Identify Annotation Data Type
Step 2: Choose Suitable Metrics
Step 3: Compute Agreement
Step 4: Interpret Scores in Context
Step 5: Act on the Results
Step 6: Report Transparently
Appendix: Summary of Inter-Annotator Agreement Metrics
Takeaways
Further Reading
References
Citation
Overview
Inter-annotator agreement (IAA) is a fundamental concept in annotation-driven research areas such as Natural Language Processing (NLP), computer vision, medical coding, and social sciences. The goal of IAA is to quantify the degree to which multiple annotators, given the same task and guidelines, produce consistent outputs.
High agreement suggests that the task is well-defined and that the annotations can be trusted as ground truth; low agreement often indicates ambiguity in the task or poor annotator guideline design.
Why Measure Inter-Annotator Agreement?
Reliability of Data
: Annotation tasks often serve as the foundation for training and evaluating supervised machine learning models. If annotations are unreliable, models trained on them will inherit that noise.
Task Clarity
: Low agreement can highlight that annotation guidelines are ambiguous, incomplete, or too subjective.
Annotator Quality
: Agreement measures can help detect inconsistent annotators or biases.
Scientific Rigor
: In empirical research, IAA serves as evidence that reported findings are reproducible and not merely artifacts of annotator idiosyncrasies.
Types of Data for Agreement
Different label/annotation data types call for different agreement metrics:
Categorical/Semantic Labels (Nominal Data)
:
Examples:
sentiment classification (positive/neutral/negative), medical diagnosis codes.
Agreement here involves checking whether annotators choose the same category.
Ordinal Labels
:
Examples:
rating scales (1–5 stars, severity levels).
Agreement must respect the fact that categories have an inherent order.
Continuous Labels
:
Examples:
bounding box coordinates in images, reaction times, or scores between 0 and 1.
Agreement is often measured via correlation or distance metrics.
Structured Outputs
:
Examples:
parse trees, dialogue act sequences, or entity spans.
Agreement requires specialized metrics that account for structured predictions.
Distributions
:
In some tasks, annotators are asked not to provide a single label, but a
probability distribution
over possible labels. This reflects uncertainty or subjectivity.
Example: In emotion annotation, one annotator may assign 0.6 probability to “joy,” 0.3 to “surprise,” and 0.1 to “neutral,” while another annotator may spread probabilities differently.
Agreement is then measured using distributional divergences such as
Total Variation distance (TV distance)
,
Kullback–Leibler divergence (KL)
, or
Jensen–Shannon divergence (JS)
:
For two discrete distributions \(P\) and \(Q\) over a label space \(\mathcal{X}\):
TV distance:
\[d_{TV}(P, Q) = \frac{1}{2} \sum_{x \in \mathcal{X}} | P(x) - Q(x) |\]
KL divergence:
\[D_{KL}(P \Vert Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}\]
JS divergence:
\[D_{JS}(P \Vert Q) = \frac{1}{2} D_{KL}\left(P \Vert M\right) + \frac{1}{2} D_{KL}\left(Q \Vert M\right), \quad M = \frac{1}{2}(P + Q)\]
These measures provide a
graded notion of disagreement
rather than a binary match/mismatch.
Classical Metrics for Inter-Annotator Agreement
Classical agreement metrics focus on categorical and ordinal labels, where the annotators assign one label per instance. They adjust for chance agreement and provide interpretable scales of reliability.
Cohen’s Kappa (\(\kappa\))
Definition
: For two annotators, Cohen’s kappa measures agreement while correcting for chance.
Formula
:
Let:
\(p_o\): observed proportion agreement
\(p_e\): expected agreement under independence
Then:
\[\kappa = \frac{p_o - p_e}{1 - p_e}\]
Suitable for
: Categorical (nominal) labels.
Use-case
: Two medical experts diagnosing patients into disease categories (yes/no cancer).
Pros
:
Corrects for chance agreement.
Easy to interpret (\(\kappa = 1\): perfect agreement, \(\kappa = 0\): chance-level).
Cons
:
Only supports two annotators.
Sensitive to class imbalance (rare categories can distort \(\kappa\)).
Scott’s Pi (\(\pi\))
Definition
: Similar to Cohen’s \(\kappa\) but uses
expected agreement under uniform distribution
rather than empirical marginals.
Formula
:
\[\pi = \frac{p_o - p_e}{1 - p_e}\]
Suitable for
: Categorical data, two annotators.
Use-case
: Two coders labeling political statements as left/right/neutral.
Pros
:
Simple and interpretable.
Historically important precursor to \(\kappa\).
Cons
:
Assumes annotators share the same distribution.
Less robust in practice than \(\kappa\).
Fleiss’ Kappa (\(\kappa\))
Definition
: Generalization of Cohen’s kappa for
more than two annotators
. Agreement is measured by comparing observed vs. expected proportions across all annotators.
Formula
:
For category \(j\):
\[P_j = \frac{1}{Nn} \sum_{i=1}^N n_{ij}\]
where \(n_{ij}\) is the number of annotators assigning category \(j\) to item \(i\).
Then compute per-item agreement and average across items.
Suitable for
: Categorical (nominal) labels with multiple annotators.
Use-case
: A crowd-sourced sentiment task with 10 annotators per review.
Pros
:
Extends Cohen’s \(\kappa\) to many annotators.
Still adjusts for chance agreement.
Cons
:
Assumes annotators are exchangeable.
Same sensitivity to imbalance issues as Cohen’s \(\kappa\).
Krippendorff’s Alpha (\(\alpha\))
Definition
: A versatile reliability coefficient that generalizes across data types.
Formula
:
\[\alpha = 1 - \frac{D_o}{D_e}\]
where:
\(D_o\) = observed disagreement
\(D_e\) = expected disagreement
The definition of disagreement depends on the data type.
Suitable for
: Categorical (nominal), ordinal, interval, and ratio data; for continuous (interval or ratio data) measurements, disagreement is computed using squared numerical differences so agreement reflects closeness of values rather than exact equality.
Use-case
: Annotators rating severity of patient symptoms on a 1–5 scale.
Pros
:
Works with any number of annotators.
Can handle missing data.
Supports various data types beyond categorical.
Cons
:
Computationally heavier than \(\kappa\).
Requires defining distance functions for non-categorical data.
Correlation-Based Measures
Unlike classical IAA metrics such as
Cohen’s \(\kappa\)
,
Scott’s \(\pi\)
,
Fleiss’ \(\kappa\)
, and
Krippendorff’s \(\alpha\)
, correlation-based measures (e.g.,
MCC
) do not explicitly model expected agreement under a chance or independence assumption. Instead, they quantify association or balance in outcomes based on observed label co-occurrences.
Correlation-based measures frame agreement through the structure of the confusion matrix (e.g., \(TP, TN, FP, FN\) in MCC), emphasizing symmetric treatment of classes and errors, whereas classical metrics conceptualize agreement as a form of inter-annotator reliability corrected for chance labeling behavior.
Classical metrics are designed to estimate annotator reliability across items and annotators, often supporting multiple annotators and label types, while correlation-based measures like
MCC
are narrower in scope—typically limited to two annotators and binary labels—and are better interpreted as measures of balanced association rather than general-purpose IAA coefficients.
Matthews Correlation Coefficient (MCC)
Definition
: Matthews Correlation Coefficient is a correlation-based measure of agreement for binary categorical annotations. In the context of inter-annotator agreement, it compares two annotators’ binary labelings using the full confusion matrix, treating agreement and disagreement symmetrically across both classes.
Formula
:
Let \(TP\), \(TN\), \(FP\), and \(FN\) denote the counts of true positives, true negatives, false positives, and false negatives between two annotators. MCC is defined as:
\[\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}\]
Range and interpretation
:
\(\mathrm{MCC} \in [-1, 1]\), where 1 indicates perfect agreement, 0 corresponds to chance-level agreement for the observed marginals, and -1 indicates perfectly inverse labeling.
Suitable for
: Binary categorical annotations, especially under class imbalance.
Use-case
: Two annotators labeling instances as positive vs. negative (e.g., toxic vs. non-toxic, disease present vs. absent).
Pros
:
Uses all four entries of the confusion matrix.
Remains informative under severe class imbalance.
Symmetric with respect to the two annotators.
Cons
:
Limited to binary labels.
Undefined when one annotator uses only a single class (degenerate marginals).
Comparative Analysis
Metric
Data Type
Use-case Example
Pros
Cons
Cohen’s \({\kappa}\)
Categorical (2 annotators)
2 doctors diagnosing disease
Adjusts for chance, easy to interpret
Only 2 annotators, sensitive to class imbalance
Scott’s \({\pi}\)
Categorical (2 annotators)
Political statement coding
Simple, historically important
Unrealistic distributional assumptions
Fleiss’ \({\kappa}\)
Categorical (many annotators)
Crowd sentiment annotation
Multi-annotator extension of Cohen’s $${\kappa}$$
Assumes annotator interchangeability
Krippendorff’s \({\alpha}\)
Categorical, ordinal, continuous (interval or ratio)
Symptom severity ratings
Versatile, supports missing data
More complex computation
Matthews Correlation Coefficient (MCC)
Binary categorical
Toxic vs. non-toxic labeling
Robust to class imbalance, uses full confusion matrix
Binary only; undefined for degenerate marginals
Why Classical Metrics Fall Short for Distributional Annotations
Most classical inter-annotator agreement (IAA) metrics—such as
Cohen’s \(\kappa\)
,
Scott’s \(\pi\)
,
Fleiss’ \(\kappa\)
, and
Krippendorff’s \(\alpha\)
—as well as correlation-based measures like the
Matthews Correlation Coefficient (MCC)
, are designed under the assumption that annotators provide a
single discrete label
per item. Under this paradigm, annotations are treated as categorical outcomes, and agreement is defined in terms of exact label matches corrected for chance.
However, this assumption increasingly fails in modern annotation settings, where labels are often
uncertain, subjective, or inherently graded
. In such cases, forcing annotators (or aggregated annotator behavior) to commit to a single label obscures meaningful information about ambiguity and disagreement.
Emergence of Distributional Annotations
With tasks involving
uncertainty, subjectivity, or ambiguity
, annotators are increasingly asked to provide a
distribution over labels
rather than a hard decision. For example:
Emotion annotation
: Annotators distribute probabilities across emotions (e.g., joy, sadness, fear).
Topic labeling
: A document may be annotated as 70% politics and 30% economics.
Crowdsourcing
: Aggregating responses from many annotators naturally yields
empirical label distributions
rather than a single consensus label.
Formally, these annotations lie in the probability simplex
\(\Delta^K = \left{ p \in \mathbb{R}^K ;\middle|; p_i \ge 0,; \sum_{i=1}^K p_i = 1 \right},\)
rather than in a finite set of categorical labels. This provides a
richer representation of annotator uncertainty and disagreement
.
Limitations of Classical Agreement Metrics
Binary versus graded disagreement
:
Metrics such as
Cohen’s \(\kappa\)
and
MCC
treat disagreement as all-or-nothing. Two annotators whose label distributions overlap substantially are treated the same as annotators whose judgments are completely opposed.
Information loss through discretization
:
Collapsing distributions to single labels (e.g., via \(\arg\max\)) discards uncertainty information and masks subtle but systematic differences between annotators.
Incompatibility with probabilistic structure
:
Classical agreement metrics assume categorical variables, not vectors constrained to the probability simplex \(\Delta^K\). As a result, they cannot operate directly on probabilistic annotations without ad hoc preprocessing.
Because of these limitations, applying classical IAA metrics to distributional data often yields misleading or impoverished assessments of agreement.
Correlation-Based Measures as Partial Workarounds
Correlation-based measures
such as
Pearson’s correlation coefficient
,
Spearman’s rank correlation coefficient
, and the
Concordance Correlation Coefficient (CCC)
are sometimes used when annotations are continuous or graded.
These measures can be applied to distributional annotations only
indirectly
, by embedding each distribution into a continuous representation—for example, via expected label values or class-wise probability vectors.
While such measures can capture
linear or monotonic association
, they do not operate on distributions as distributions. In particular, they ignore the geometry of the probability simplex and fail to provide a principled notion of distance between probability mass assignments.
As a result, correlation-based measures should be viewed as
association metrics on derived quantities
, not true distributional agreement measures.
Pearson’s Correlation Coefficient (\(r\))
Definition
: Pearson’s correlation coefficient measures the strength and direction of a linear relationship between two sets of continuous annotations. In inter-annotator agreement, it captures whether annotators vary together in a linear fashion, regardless of absolute scale alignment.
Formula
:
\[r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\sqrt{\sum_i (y_i - \bar{y})^2}}\]
Range and interpretation
: \(r \in [-1, 1]\), where 1 indicates perfect positive linear association, 0 no linear association, and -1 perfect negative linear association.
Suitable for
: Interval or ratio-scale continuous annotations.
Use-case
: Two annotators assigning real-valued emotion intensity scores or regression-style labels.
Pros
:
Simple and widely understood.
Works naturally with continuous-valued data.
Interpretable as linear association strength.
Cons
:
Does not measure agreement in absolute values (only association).
Insensitive to systematic bias in mean or scale.
Sensitive to outliers.
Spearman’s Rank Correlation Coefficient (\(\rho\))
Definition
: Spearman’s rank correlation coefficient measures the strength and direction of a monotonic relationship between two sets of annotations. It operates on the ranks of the values rather than the raw values, making it suitable when only the ordering of annotations matters.
Formula
:
Let \(R(x_i)\) and \(R(y_i)\) denote the ranks of \(x_i\) and \(y_i\). Spearman’s \(\rho\) is defined as Pearson’s correlation applied to the ranks:
\[\rho = \frac{\sum_i (R(x_i) - \overline{R(x)})(R(y_i) - \overline{R(y)})}{\sqrt{\sum_i (R(x_i) - \overline{R(x)})^2}\sqrt{\sum_i (R(y_i) - \overline{R(y)})^2}}\]
For the special case of no tied ranks, this simplifies to:
\[\rho = 1 - \frac{6\sum_i d_i^2}{n(n^2 - 1)}\]
where \(d_i\) is the difference between ranks.
Range and interpretation
: \(\rho \in [-1, 1]\), with interpretations analogous to Pearson’s \(r\), but in terms of monotonic rather than strictly linear relationships.
Suitable for
: Ordinal data, continuous data where only relative ordering is meaningful.
Use-case
: Annotators ranking items by sentiment strength or perceived relevance.
Pros
:
Robust to monotonic nonlinear relationships.
Less sensitive to outliers than Pearson’s \(r\).
Appropriate for ordinal scales.
Cons
:
Ignores absolute differences between annotation values.
Still measures association, not absolute agreement.
Concordance Correlation Coefficient (CCC)
Definition
: The Concordance Correlation Coefficient measures agreement between two sets of continuous annotations by jointly assessing precision and accuracy. It extends Pearson’s correlation by penalizing systematic differences in location (mean) and scale (variance), thereby measuring how closely the annotators’ scores align with the identity line \(y = x\).
Formula
:
Let \(x_i\) and \(y_i\) denote the annotations from two annotators, with means \(\mu_x, \mu_y\), variances \(\sigma_x^2, \sigma_y^2\), and covariance \(\sigma_{xy}\). CCC is defined as:
\[\rho_c = \frac{2\sigma_{xy}}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}\]
Equivalently, CCC can be written as:
\[\rho_c = \rho \cdot C_b\]
where \(\rho\) is Pearson’s correlation coefficient and \(C_b\) is a bias correction factor capturing differences in mean and scale.
Range and interpretation
:
\(\rho_c \in [-1, 1]\), where 1 indicates perfect agreement (identical values), 0 indicates no agreement, and -1 indicates perfect inverse agreement.
Suitable for
: Continuous interval or ratio-scale annotations where absolute agreement matters.
Use-case
: Annotators assigning continuous scores such as emotion intensity, medical measurements, or regression targets.
Pros
:
Measures agreement rather than mere association.
Penalizes systematic bias between annotators.
More appropriate than Pearson’s \(r\) when scale alignment matters.
Cons
:
Typically defined for two annotators (multi-annotator use requires pairwise aggregation).
Sensitive to outliers, similar to Pearson’s correlation.
When Correlation-Based Measures Are Still Appropriate
Correlation-based measures
—including
Pearson’s \(r\)
,
Spearman’s \(\rho\)
, and
CCC
—can still play a useful role as
secondary analyses
alongside distributional divergences, provided their limitations are clearly understood.
These measures are appropriate when
distributional annotations are reduced to continuous summaries
, such as:
Expected label values (e.g., \(\mathbb{E}[y] = \sum_k k p_k\) for ordinal labels),
Aggregate intensity scores,
Class-wise probability vectors analyzed dimension by dimension.
In such cases:
Pearson’s \(r\)
is useful for assessing whether annotators’ derived scores vary together linearly, even if they differ in scale.
Spearman’s \(\rho\)
is appropriate when only relative ordering matters, such as ranking items by perceived intensity or relevance.
CCC
is preferred when
absolute agreement
on continuous-valued summaries is important and systematic bias should be penalized.
However, correlation-based measures should not be interpreted as
distributional agreement metrics
. They do not operate on probability distributions directly, ignore the geometry of the probability simplex, and can obscure meaningful differences in probability mass allocation.
Best practice is therefore to use
distributional divergences
(e.g., TV, KL, JS) as the primary measures of agreement, with
correlation-based metrics
serving as complementary diagnostics on derived continuous representations.
If you want, the next step could be a brief unifying guideline section on choosing between divergence-based and correlation-based agreement metrics depending on annotation design.
Transition to Divergence-Based Measures
Rather than measuring categorical agreement or scalar association, distributional approaches quantify agreement by computing
distances or divergences between probability distributions
.
These measures allow us to assess not only whether annotators disagree, but
how far apart their probability judgments are
.
This motivates the use of
distributional agreement metrics
such as:
Total Variation (TV) distance
Kullback–Leibler (KL) divergence
Jensen–Shannon (JS) divergence
Distributional Agreement Metrics
When annotators provide
probability distributions
instead of single labels, agreement must be measured by comparing two distributions directly. Metrics designed for this purpose are typically referred to as
distances
or
divergences
on the probability simplex.
Unlike classical or correlation-based measures, these metrics operate natively on distributions \(P, Q \in \Delta^K\), preserving information about
uncertainty, partial agreement, and graded disagreement
.
Distributional agreement metrics answer questions such as:
How much probability mass do two annotators assign differently?
How costly is it, in information-theoretic terms, to approximate one annotator’s beliefs with another’s?
Are two distributions close in a symmetric and bounded sense?
Below, we describe three widely used distributional agreement metrics—
Total Variation distance
,
Kullback–Leibler divergence
, and
Jensen–Shannon divergence
—each of which captures a different notion of disagreement between probability distributions.
Total Variation (TV) Distance
Definition
: For two discrete distributions \(P\) and \(Q\) over a label set \(\mathcal{X}\), the
Total Variation distance
quantifies the largest possible difference between the probabilities assigned by \(P\) and \(Q\) to the same event. Intuitively, it measures how much probability mass must be reallocated to transform one distribution into the other.
Formula
:
\[d_{TV}(P, Q) = \frac{1}{2} \sum_{x \in \mathcal{X}} \left| P(x) - Q(x) \right|\]
Intuition
: TV distance measures the maximum absolute discrepancy in probability mass across categories. It can be interpreted as the largest possible difference in probability that the two annotators assign to the same outcome.
Range
: \([0, 1]\)
\(0\) indicates identical distributions.
\(1\) indicates completely disjoint support.
Suitable for
: Any form of distributional annotation, regardless of label semantics.
Use-case
: Comparing how two annotators distribute probability mass across emotions for a sentence.
Pros
:
Symmetric
and
bounded
.
A true
metric
(satisfies the triangle inequality).
Highly
interpretable
.
Cons
:
Ignores information-theoretic structure.
Can be relatively coarse in high-dimensional label spaces.
Kullback–Leibler (KL) Divergence
Definition
: The
Kullback–Leibler (KL) divergence
quantifies how one probability distribution \(Q\) diverges from another distribution \(P\). It measures the expected number of extra bits required to encode samples drawn from \(P\) using a code optimized for \(Q\). KL divergence is rooted in information theory and is inherently
asymmetric
.
Formula
:
\[D_{KL}(P \Vert Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}\]
Intuition
: KL divergence measures the inefficiency incurred when \(Q\) is used as a surrogate for \(P\). Large values indicate that \(Q\) assigns low probability to events that \(P\) considers likely.
Range
: \([0, \infty)\)
\(0\) if and only if \(P = Q\).
Suitable for
: Distributional annotations where one distribution can be treated as a reference or target and the other as an approximation.
Use-case
: Quantifying how much information is lost if one annotator’s probability distribution is used to approximate another’s.
Pros
:
Strong
information-theoretic interpretation
.
Highly sensitive to discrepancies in
low-probability events
.
Cons
:
Asymmetric
: \(D_{KL}(P \Vert Q) \neq D_{KL}(Q \Vert P)\).
Undefined
when \(Q(x) = 0\) and \(P(x) > 0\).
Unbounded and often
hard to interpret numerically
.
Jensen–Shannon (JS) Divergence
Definition
: The
Jensen–Shannon (JS) divergence
is a symmetrized and smoothed variant of the KL divergence. It measures how much each of two distributions diverges, on average, from their mean distribution. Unlike KL divergence, JS divergence is always finite and symmetric.
Let the mean distribution be:
\[M = \frac{1}{2}(P + Q).\]
Formula
:
\[D_{JS}(P \Vert Q) = \frac{1}{2} D_{KL}(P \Vert M) + \frac{1}{2} D_{KL}(Q \Vert M).\]
Intuition
: JS divergence captures the average information loss when both distributions are approximated by their midpoint. It balances sensitivity to differences with numerical stability.
Range
: \([0, \log 2]\)
Often normalized to \([0, 1]\) for convenience.
\(0\) indicates identical distributions.
Suitable for
: Any pair of probability distributions, especially when
symmetry and stability
are desired.
Use-case
: Comparing how two annotators spread probability mass across multiple labels while avoiding undefined or infinite values.
Pros
:
Symmetric
and
always finite
.
The square root \(\sqrt{D_{JS}}\) defines a true
metric
.
More
robust in practice
than KL divergence.
Cons
:
Less directly interpretable than Total Variation distance.
Still sensitive to smoothing and normalization choices.
Comparative Analysis
Correlation Measures: Pearson’s \(r\), Spearman’s \(\rho\), and Concordance Correlation Coefficient (CCC)
Metric
Data Type
Use-case Example
Pros
Cons
Pearson’s \(r\)
Continuous (interval or ratio)
Emotion intensity or regression-style scores
Simple and widely understood; captures linear co-variation
Measures association rather than absolute agreement; insensitive to mean or scale bias; sensitive to outliers
Spearman’s \(\rho\)
Ordinal or continuous
Ranking sentiment strength or perceived relevance
Captures monotonic relationships; robust to nonlinear scaling and outliers
Ignores absolute differences; measures ordering rather than agreement
Concordance Correlation Coefficient (CCC)
Continuous (interval or ratio)
Medical measurements or continuous annotation tasks
Measures absolute agreement; penalizes systematic bias in mean and scale
Primarily defined for two annotators; sensitive to outliers
Divergence Measures: TV, KL, and JS Divergence
Metric
Symmetric?
Range
Interpretability
Sensitivity
Use-case example
Total Variation distance
Yes
$$[0,1]$$
Very high (maximum probability difference)
Treats all categories equally
Emotion probability distributions
KL divergence
No
$$[0,\infty)$$
Information-theoretic, less intuitive
Highly sensitive to rare events
Approximation error analysis
JS divergence
Yes
$$[0,\log 2]$$
Balanced, bounded, metric via $$\sqrt{D_{JS}}$$
Smoothed, avoids infinities
General distributional IAA
Practical Considerations for Inter-Annotator Agreement
IAA analysis requires more than just selecting a formula — it involves understanding the
data type, annotation context, and interpretability needs
. This section provides guidance on how to choose metrics, what to watch out for, and how to interpret agreement scores.
Choosing a Metric by Data Type
Data Type
Recommended Metrics
Notes
Categorical (nominal)
Cohen’s \(\kappa\) (2 annotators), Fleiss’ \(\kappa\) (many), Krippendorff’s \(\alpha\)
Must check for class imbalance effects
Ordinal
Weighted Cohen’s \(\kappa\), Krippendorff’s \(\alpha\), Spearman’s \(\rho\)
Use distance-based weighting to respect ordering
Continuous
Pearson’s \(r\), Intraclass Correlation (ICC), Krippendorff’s \(\alpha\)
Handle outliers carefully, scale-sensitive
Structured outputs
Task-specific metrics (e.g., overlap F1, span-based agreement)
Define what counts as “match” structurally
Distributions
TV distance, JS divergence, KL divergence
Do not collapse to argmax labels; keep full distributions
Rule of thumb
:
If annotators give single discrete labels \(\rightarrow\) use chance-corrected categorical metrics.
If annotators give scores or ranks \(\rightarrow\) use correlation- or distance-based measures.
If annotators give full probability distributions \(\rightarrow\) use divergence measures.
Interpreting Agreement Levels
There is no absolute scale, but a commonly used heuristic (adapted from
Landis & Koch, 1977
interpretation for Cohen’s \(\kappa\)) is:
Agreement value
Interpretation
\(< 0.0\)
Worse than chance
\(0.0–0.20\)
Slight agreement
\(0.21–0.40\)
Fair agreement
\(0.41–0.60\)
Moderate agreement
\(0.61–0.80\)
Substantial agreement
\(0.81–1.00\)
Almost perfect
For
divergence metrics
(TV, KL, JS), lower values mean closer distributions. Typical observed ranges:
TV distance:
< 0.1 \(\rightarrow\) very similar; > 0.3 \(\rightarrow\) strong disagreement
JS divergence:
< 0.05 \(\rightarrow\) close; > 0.2 \(\rightarrow\) widely different
KL divergence:
highly variable; compare relative changes, not absolute cutoffs.
Handling Annotator Bias and Class Imbalance
Class imbalance
can inflate or deflate \(\kappa\)-like metrics. Consider reporting class distributions alongside agreement.
Annotator bias
(systematic skew) can lower \(\kappa\) even if raw agreement is high.
Consider using
confusion matrices
to inspect which categories cause disagreement.
Missing Data and Sparse Annotations
Krippendorff’s \(\alpha\)
is robust to missing annotations and is the safest choice for incomplete data.
For divergence-based measures, ensure smoothing (e.g., add small \(\epsilon\)) to avoid zeros that break KL.
Computational Considerations
\(\kappa\)-type metrics
are computationally cheap (matrix counts).
Krippendorff’s \(\alpha\)
is \(O(N \times A^2)\) for \(N\) items and \(A\) annotators — still feasible but heavier.
Divergence-based metrics
are \(O(K)\) per pair of distributions, where \(K\) is the number of categories.
If annotator sets are large, prefer efficient
pairwise sampling
strategies or
aggregate distributions
.
Mathematical Relationships Between TV, KL, and JS
While
Total Variation Distance
,
Kullback–Leibler divergence
, and
Jensen–Shannon divergence
measure different aspects of distributional difference, they are connected through known inequalities. Understanding these links helps interpret and compare their values meaningfully.
Pinsker’s Inequality (KL vs. TV)
Pinsker’s inequality provides an upper bound on TV distance in terms of KL divergence:
\[d_{TV}(P, Q) \le \sqrt{\frac{1}{2} D_{KL}(P \Vert Q)}\]
This means:
If KL divergence is small, then TV must also be small.
Small KL implies distributions are close in absolute terms.
However, small TV does
not
guarantee small KL (KL can blow up when \(Q(x) \approx 0\)).
Implication
:
KL is more sensitive to low-probability mismatches than TV.
Lower Bound on KL via TV
A lesser-known inequality gives a lower bound on KL in terms of TV:
\[D_{KL}(P \Vert Q) \ge 2 d_{TV}^2(P, Q)\]
This shows that large TV implies large KL. Together with Pinsker’s inequality, this pins down their growth relationship:
\[2 d_{TV}^2(P,Q) \le D_{KL}(P \Vert Q)\]
JS Divergence Related to KL and TV
JS divergence is defined as:
\[D_{JS}(P|Q) = \frac{1}{2}D_{KL}(P \Vert M) + \frac{1}{2}D_{KL}(Q \Vert M), \quad M = \frac{1}{2}(P+Q)\]
and satisfies:
\[D_{JS}(P \Vert Q) \le \log 2\]
It inherits KL’s information-theoretic basis while being symmetric and bounded.
Also, it relates to TV as:
\[d_{TV}^2(P,Q) \le \frac{1}{2} D_{JS}(P \Vert Q)\]
So:
JS grows at least as fast as \(d_{TV}^2\).
JS is upper-bounded, while KL is unbounded.
JS is often preferred for
interpretability and numerical stability
.
Comparative Analysis of Theoretical Relationships
Pair
Inequality
Interpretation
TV vs. KL
\(d_{TV} \le \sqrt{\tfrac{1}{2} D_{KL}}\)
Small KL implies small TV
TV vs. KL
\(2 d_{TV}^2 \le D_{KL}\)
Large TV implies large KL
TV vs. JS
\(d_{TV}^2 \le \tfrac{1}{2} D_{JS}\)
JS lower bounds TV squared
JS vs. KL
\(D_{JS} \le D_{KL} \quad \text{(if } P=Q \text{ support)}\)
JS is smoothed and bounded version of KL
Key takeaways
:
TV gives an absolute probability difference.
KL gives a relative (log-based) penalty, very sensitive to rare events.
JS sits between them: symmetric, smoothed, and bounded, making it ideal for
practical agreement comparisons
.
Putting It All Together: A Workflow for Measuring IAA
This section provides a
step-by-step pipeline
for measuring inter-annotator agreement, choosing the correct metric, and interpreting the results in context.
Step 1: Identify Annotation Data Type
Before picking any metric, classify your annotation outputs into one of these types:
Categorical (nominal):
single class per item, no order
Ordinal:
discrete ranks with meaningful order
Continuous:
numeric values on a scale (say, interval or ratio data)
Structured:
spans, trees, sequences
Distributions:
full probability vectors over categories
Tip
: If annotators are uncertain and spread probability mass, treat their outputs as distributions rather than forcing hard labels.
Here is the
updated version
with
Scott’s \(\pi\)
,
MCC
, and
Total Variation distance
added in a principled way, while keeping the same formatting and scope.
Step 2: Choose Suitable Metrics
Use this quick mapping:
Data Type
Recommended Metrics
Categorical
Cohen’s \(\kappa\) (2), Scott’s \(\pi\) (2), Fleiss’ \(\kappa\) (many), Krippendorff’s \(\alpha\), Matthews Correlation Coefficient (MCC, binary)
Ordinal
Weighted Cohen’s \(\kappa\), Krippendorff’s \(\alpha\), Spearman’s \(\rho\)
Continuous
Pearson’s \(r\), Intraclass Correlation (ICC), Concordance Correlation Coefficient (CCC), Krippendorff’s \(\alpha\)
Structured
Task-specific matching (span F1, overlap measures)
Distributions
Total Variation distance (\(d_{TV}\)), Kullback–Leibler divergence (\(D_{KL}\)), Jensen–Shannon divergence (\(D_{JS}\)), Earth Mover’s Distance (\(EMD\))
Guidelines
:
If you care primarily about
agreement beyond chance
, use \(\kappa\)-type metrics (including Scott’s \(\pi\) and Krippendorff’s \(\alpha\)).
If labels are
binary and imbalanced
, consider
MCC
as a complementary association-based diagnostic.
If you care about
numerical closeness or uncertainty
, use
correlation-based measures
(for continuous summaries) or
distributional divergences
(for probability distributions).
Step 3: Compute Agreement
Clean data: handle missing annotations, standardize label sets.
For categorical metrics, build an item × annotator label matrix.
For distributional metrics, build an item × annotator probability matrix.
Compute:
Pairwise agreement (between annotator pairs)
Average agreement (overall reliability)
Tip
: For large numbers of annotators, use random subsampling of pairs to reduce computation.
Step 4: Interpret Scores in Context
Compare against known benchmarks (e.g., \(\kappa\) > 0.6 is substantial agreement).
For divergence metrics:
\(d_{TV} < 0.1\) or \(D_{JS} < 0.05\) \(\rightarrow\) very high agreement
\(d_{TV} > 0.3\) or \(D_{JS} > 0.2\) \(\rightarrow\) strong disagreement
Visualize distributions and confusion matrices to identify where disagreements occur.
Important
: Absolute cutoffs are less meaningful than
relative comparisons across tasks or iterations
.
Step 5: Act on the Results
If agreement is low:
Refine annotation guidelines
Provide more training/examples to annotators
Identify and retrain or remove inconsistent annotators
If agreement is high:
Proceed with data aggregation and model training
Optionally, use annotator reliability as weights in aggregation
Step 6: Report Transparently
When publishing or sharing results:
Specify which metric you used and why.
Report number of annotators, number of samples, and how missing data was handled.
Include both agreement values and class distributions for context.
Appendix: Summary of Inter-Annotator Agreement Metrics
Metric
Data Type
Formula
Interpretation
Pros
Cons
Typical Range / Use-Case
Cohen’s \(\kappa\)
Categorical (2 annotators)
\(\kappa = \frac{p_o - p_e}{1 - p_e}\)
Agreement beyond chance between two annotators
Adjusts for chance; simple
Only two annotators; sensitive to class imbalance
\([0, 1]\); medical diagnoses, binary coding
Scott’s \(\pi\)
Categorical (2)
\(\pi = \frac{p_o - p_e}{1 - p_e}\) with uniform expected \(p_e\)
Chance-corrected agreement with equal priors
Simple, historic
Unrealistic distribution assumption
\([0, 1]\); political or sentiment coding
Fleiss’ \(\kappa\)
Categorical (many annotators)
Mean chance-corrected agreement across annotators
Multi-annotator extension of \(\kappa\)
Handles multiple annotators
Assumes annotators are interchangeable; imbalance sensitive
\([0, 1]\); crowdsourced labeling
Krippendorff’s \(\alpha\)
Nominal → ratio
\(\alpha = 1 - \frac{D_o}{D_e}\)
General reliability across data types
Works with missing data; flexible
More complex computation
\([0, 1]\); mixed data, psychological scales
Weighted \(\kappa\)
Ordinal
Weighted form of \(\kappa\) with penalty matrix \(w_{ij}\)
Agreement respecting order of categories
Considers ordinal distances
Needs chosen weights; subjective
\([0, 1]\); rating scales, quality scores
Pearson’s \(r\)
Continuous
\(r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\sqrt{\sum_i (y_i - \bar{y})^2}}\)
Linear correlation of scores
Interpretable; handles continuous values
Sensitive to outliers; only linear
\([-1, 1]\); numeric scoring, regression tasks
Spearman’s \(\rho\)
Ordinal / continuous
Correlation of rank orders
Monotonic relationship between annotators
Order-based, robust
Ignores exact scale differences
\([-1, 1]\); ranking tasks
Intraclass Corr. (ICC)
Continuous
Variance ratio model
Consistency among several raters
Captures group consistency
Depends on model assumptions
\([0, 1]\); behavioral, clinical studies
TV distance
Distributions
\(d_{TV}(P,Q)=\tfrac{1}{2}\sum_x |P(x)-Q(x)|\)
Max difference in probability mass
Bounded, symmetric, metric
Ignores info-theoretic nuance
\([0, 1]\); probabilistic emotion or topic labels
KL divergence
Distributions
\(D_{KL}(P \Vert Q)=\sum_x P(x)\log \tfrac{P(x)}{Q(x)}\)
Information loss using \(Q\) for \(P\)
Info-theoretic; sensitive to rare events
Asymmetric; undefined for zeros
\([0, \infty)\); model approximation error
JS divergence
Distributions
\(D_{JS}(P|Q)=\tfrac{1}{2}D_{KL}(P \Vert M)+\tfrac{1}{2}D_{KL}(Q \Vert M),\)
\(M=\tfrac{1}{2}(P+Q)\)
Smoothed, symmetric version of KL
Symmetric; bounded; interpretable
Still needs smoothing
\([0, \log 2]\); general probabilistic agreement
Task-specific overlap (\(F_1\), span \(F_1\))
Structured outputs
\(F_1=\frac{2PR}{P+R}\)
Overlap or matching agreement
Intuitive for structured data
Needs domain-specific definition
\([0, 1]\); entity extraction, segmentation
Takeaways
Symmetry
:
TV and JS are symmetric; KL is not.
Boundedness
:
\(d_{TV} \in [0, 1], \quad D_{JS} \in [0, \log 2], \quad D_{KL} \in [0, \infty)\)
Data completeness
:
Krippendorff’s \(\alpha\) handles missing data best.
When in doubt
:
For categorical labels \(\rightarrow\) Cohen/Fleiss \(\kappa\).
For continuous or ordinal \(\rightarrow\) correlation or \(\alpha\).
For distributions \(\rightarrow\) \(d_{TV}\) or \(D_{JS}\) divergence.
Further Reading
Inter-coder Agreement for Computational Linguistics
A Coefficient of Agreement for Nominal Scales
Measuring Nominal Scale Agreement Among Many Raters
Reliability of Content Analysis: The Case of Nominal Scale Coding
Content Analysis: An Introduction to Its Methodology
Computing Krippendorff’s Alpha-Reliability
On Krippendorff’s Alpha Coefficient (Revised 2015)
DKPro Agreement: A Java Library for Measuring Inter-Rater Agreement
An Elementary Introduction to Information Geometry
Elements of Information Theory
Measuring Agreement on Set-valued Items (MASI) for Semantic and Pragmatic Annotation
Semeval-2016 Task 5: Aspect Based Sentiment Analysis
Reliability in Software Engineering Qualitative Research Using Krippendorff’s Alpha and Atlas.ti
krippendorffsalpha: An R Package for Measuring Agreement Using Krippendorff’s Alpha Coefficient
The Measurement of Observer Agreement for Categorical Data
References
Survey Article: Inter-Coder Agreement for Computational Linguistics (Artstein & Poesio, 2008)
A Coefficient of Agreement for Nominal Scales (Cohen, 1960)
Inter-Coder Agreement — Direct MIT/COLI PDF
A Brief Tutorial on Inter-Rater Agreement (DKPro / Meyer)
Chapter on Agreement Coefficients for Nominal Ratings (AgreeStat PDF)
Krippendorff’s alpha — Wikipedia page
Reliability in Software Engineering Qualitative Research using Krippendorff’s α (arXiv preprint)
Citation
@article{Chadha2020DistilledInterAnnotatorAgreement,
  title   = {Inter-Annotator Agreement},
  author  = {Chadha, Aman and Jain, Vinija},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}