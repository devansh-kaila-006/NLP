# Inductive Bias

**Source**: https://aman.ai/primers/ai/inductive-bias
**Category**: Model Architecture
**Word Count**: 2262

---

Background
Inductive bias as a guide to model exploration
Choice of algorithm determines inductive bias
Feature engineering
Understanding the data
Inductive biases of standard neural architectures
Why should you make inductive biases in models? What can’t we consider the whole search space?
Citation
Background
What makes working on new machine learning (ML) use cases so exciting, and at times so frustrating, is ML’s lack of hard and fast rules. A few aspects of the model development process can be codified; for example, data should always be separated into strictly disjoint training and test sets to ensure that model performance isn’t attributable to overfitting. But at the heart of machine learning prototyping is a heavy dose of guesswork and intuition. What algorithm, or what representation of the data, will yield the most effective predictions? Once a candidate set of models is generated, evaluating them and selecting the best can be a (fairly) straightforward process. But the framework for generating the models is not nearly as clear.
For the more challenging ML-based projects, the mission is to develop machine learning models to extract critical data points from messy, unstructured records. We’ve found that the path to prototyping high-performing models is clearer when using a foundational ML concept called inductive bias as a guide.
Inductive bias as a guide to model exploration
Inductive reasoning is the process of learning general principles on the basis of specific instances – in other words, it’s what any machine learning algorithm does when it produces a prediction for any unseen test instance on the basis of a finite number of training instances. Inductive bias describes the tendency for a system to prefer a certain set of generalizations over others that are equally consistent with the observed data.
The term bias gets a bad rap, and it’s indeed a big problem when societal biases sneak into algorithmic predictions. But inductive bias is absolutely essential to machine learning (and human learning, for that matter). Without inductive bias, a learner can’t generalize from observed examples to new examples better than random guessing.
To see why, we can imagine an unbiased algorithm attempting to generalize, as described in this classic paper. Suppose an unbiased algorithm is trying to learn a mapping from numbers in the range one to 100 to the labels TRUE and FALSE. It observes that two, four and six are labeled TRUE, while one, three and five are labeled FALSE. What is the label for seven? To the unbiased learner, there are 2^(94) possible labelings of the numbers seven through 100, all equally possible, with seven labeled TRUE in one half and seven labeled FALSE in the other. Thus the algorithm is reduced to guessing. To a human, of course, there is a clear pattern of even numbers labeled TRUE and odds labeled FALSE, so a label of FALSE for seven seems likely. This type of belief, that certain hypotheses like “evens are labeled TRUE” seem likely, and other hypotheses like “evens are labeled TRUE up through 6, and the rest of the labeling is random” seem unlikely, is inductive bias. This is what makes both human learning and machine learning possible.
Once we know that inductive bias is needed, the obvious question is, “what bias is best?” The no free lunch theorem of machine learning shows that there is no one best bias, and that whenever algorithm A outperforms algorithm B in one set of problems, there is an equally large set of problems in which algorithm B outperforms algorithm A (even if algorithm B is randomly guessing!). This means that finding the best model for a machine learning problem isn’t about looking for one “master algorithm,” but about finding an algorithm with biases that make sense for the problem being solved.
Choice of algorithm determines inductive bias
Every ML algorithm used in practice, from nearest neighbors to gradient boosting machines, comes with its own set of inductive biases about what classifications (or functions, in the case of a regression problem) are easier to learn. Nearly all learning algorithms are biased to learn that items similar to each other (“close” to one another in some feature space) are more likely to be in the same class. Linear models, such as logistic regression, additionally assume that the classes can be separated by a linear boundary (this is a “hard” bias, since the model can’t learn anything else). For regularized regression, which is the type almost always used in ML, there’s an additional bias towards learning a boundary that involves few features, with low feature weights (this is a “soft” bias because the model can learn class boundaries involving lots of features with high weights, it’s just harder/requires more data).
Even deep learning models have inductive biases, and these biases are part of what makes them so effective. For example, an LSTM (long short-term memory) neural network is effective for natural language processing tasks because it is biased towards preserving contextual information over long sequences.
As an ML practitioner, knowledge about the domain and difficulty of your problem can help you choose the right algorithm to apply. As an example, let’s take the problem of identifying whether a patient has been diagnosed with metastatic cancer by extracting relevant terms from clinical notes. In this case, logistic regression performs well because there are many independently informative terms (‘mets,’ ‘chemo,’ etc.). For other problems, like extracting the result of a genetic test from a complex PDF report, we get better performance using LSTMs that can account for the long-range context of each word.
Once a base algorithm has been chosen, understanding its biases can also help you perform feature engineering, the process of choosing which information to input into the learning algorithm.
Feature engineering
It can be tempting to input as much information as possible into the model, on the reasonable assumption that providing more information will allow the model to uncover additional patterns. But adding too many features can cause overfitting and performance degradation, particularly in non-consumer tech applications where labeled data is scarce. Sometimes providing less information, or transforming existing information, better aligns the data with the model’s inductive biases and yields better results.
For instance, in clinical notes, providers often copy forward text from an old note to a new note; some electronic health record systems even do this automatically. This means that the number of times a given word appears isn’t necessarily indicative of how many times the provider actually typed it, or how important it is. Because of this, inputting word counts or TF-IDF representations of each word into a linear model, where a higher value is necessarily linked to a greater effect on the prediction, is ineffective. Instead, we’ve often found better results from simply using a binary representation of each word, indicating if it occurred at all or not.
A second example involves the extraction of a genetic test result from scanned documents across many labs. One important piece of information (among several) in the reports is a raw percentage score ranging from 0% to 100%. However, our model’s task is a rougher classification into high and low expression levels. By using regular expressions we convert percentages in the range of 1% to 50% into a consistent “
lt50pct
” token, and higher percentages into a “
gt50pct
” token. This conversion adds no new information, but significantly increases the performance of a regularized linear model. This is because the model no longer has to learn 100 separate coefficients for 1% through 100%, but now only has to learn two, lessening the noise in the training process and aligning with the model’s inductive bias to learn few coefficients with high-magnitude values.
Understanding the data
There is no set procedure for finding the inductive biases that will yield the best learning for a new ML problem, nor is there a set way to ensure they are present in the chosen algorithm. But the key is to understand the human-perceivable structures and patterns in the problem, and think through how and whether they can be encoded through feature engineering and learned by the model.
To achieve this, there’s no replacement for sitting down with subject matter experts to gain their perspective on the problem and understand how they would approach the prediction task at hand (keeping in mind, of course, that ML algorithms often do find patterns beyond those perceived by humans). A next step is to spend time doing exploratory data analysis, and error analysis once an initial model has been trained. A good idea is to spend time partnering with domain experts to analyze documents that a model misclassifies, in order to understand the kinds of linguistic patterns the model may be missing or being confused by. Looking inside the current model itself, to examine coefficients or feature importance values, can also be an effective way to understand what the model is currently learning, and what it is not learning that you would like it to.
Machine learning can seem at first to offer a way out of having to use human intuition and inferences to crack a prediction problem. And ML can, in fact, achieve feats of inductive learning that are plainly impossible for a human. But as research has shown, there is no one algorithm that performs well on all tasks, nor is there one that can simply be applied to any problem without a second thought. Instead, human intuition, domain knowledge, and understanding of the inductive biases inherent in each algorithm are still needed to fit an ML method to the problem at hand.
Even with the soundest reasoning for a change in algorithm or features, machine learning prototyping is still laden with many more failures than successes. But approaching feature engineering and algorithm choice as an attempt to match the biases of the model to the structure of the problem can be a powerful tool to move model development in the right direction.
Inductive biases of standard neural architectures
Every model architecture has an inherent inductive bias which helps understand patterns in data and thus enables learning. For instance, CNNs exhibit spatial parameter sharing, translational/spatial invariance, while RNNs exhibit temporal parameter sharing.
Relational inductive biases, deep learning, and graph networks
by Battaglia et al. (2018) from DeepMind/Google, MIT and the University of Edinburgh offers a great overview of the relational inductive biases of various neural net architectures, summarized in the table below from the paper.
YouTube Video from UofT CSC2547: Relational inductive biases, deep learning, and graph networks
;
Slides by KAIST on inductive biases, graph neural networks,
attention and relational inference
Why should you make inductive biases in models? What can’t we consider the whole search space?
Inductive biases in machine learning models refer to the set of assumptions the model makes about the underlying pattern it’s trying to learn from the data. These biases guide the learning algorithm to prefer certain solutions over others. Implementing inductive biases is essential for several reasons, and there are practical limitations to considering the entire search space in machine learning tasks.
Why Inductive Biases are Necessary:
Feasibility of Learning:
Without any inductive biases, a learning algorithm would not be able to generalize beyond the training data because it would have no preference for simpler or more probable solutions over more complex ones. In the absence of inductive biases, the model could fit the training data perfectly but fail to generalize to new, unseen data (overfitting).
Curse of Dimensionality:
As the dimensionality of the input space increases, the amount of data needed to ensure that all possible combinations of features are well-represented grows exponentially. Inductive biases help to reduce the effective dimensionality or the search space, making learning feasible with a realistic amount of data.
No Free Lunch Theorem:
This theorem states that no single learning algorithm is universally better than others when averaged over all possible problems. Inductive biases allow algorithms to specialize, performing better on a certain type of problem at the expense of others.
Computational Efficiency:
Exploring the entire hypothesis space is often computationally infeasible, especially for complex problems. Biases help reduce the search space, making training more computationally efficient.
Incorporating Domain Knowledge:
Inductive biases can be a way to inject expert knowledge into the model, allowing it to learn more efficiently and effectively. For example, convolutional neural networks are biased towards image data due to their architectural design, which is suited for spatial hierarchies in images.
Limitations of Exploring the Whole Search Space:
Computational Constraints:
The size of the complete hypothesis space for even moderately complex models can be astronomically large, making it computationally impossible to explore thoroughly.
Risk of Overfitting:
Without biases, models are more likely to fit noise in the training data, leading to poor generalization.
Data Limitations:
In practice, we have limited data. Without biases guiding the learning process, the amount of data required to learn meaningful patterns would be impractically large.
Interpretability and Simplicity:
Models learned without biases tend to be more complex and harder to interpret. Simpler models (encouraged by appropriate biases) are often preferred because they are easier to understand, debug, and validate.
Conclusion
:
In summary, inductive biases in machine learning models are crucial for guiding the learning process, making it computationally feasible, and ensuring that models generalize well to new, unseen data. These biases are a response to practical limitations in data availability, computational resources, and the inherent complexity of learning tasks.
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DistilledInductiveBias,
  title   = {Inductive Bias},
  author  = {Chadha, Aman},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}