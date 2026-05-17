# Multiclass vs. Multilabel Classification

**Source**: https://aman.ai/primers/ai/multiclass-vs-multilabel-classification
**Category**: Miscellaneous
**Word Count**: 746

---

Multi-class classification
Multi-label classification
Examples
Graphical interpretation
Multi-class classification
Multi-label classification
Multilabel Multi-class classification
Multi-class vs. Multi-label Classification
References
Citation
Multi-class classification
Multi-class classification is a type of classification task with
more than two
classes, for e.g., classifying a set of images of fruits which may be oranges, apples, or pears. Multi-class classification makes the assumption that each sample is assigned to
one and only one
label: a fruit can be either an apple or a pear but not both at the same time. In other words, multi-class classification assumes that the labels are mutually exclusive.
Some examples of multi-class classification problems are sentiment classification, spam detection, etc.
Multi-class classification can follow two different strategies as outlined below (image taken from
Atmosera: Multiclass Classification
):
Note that for binary classification, one-vs-one and one-vs-all are equivalent (since there are only two classes).
Softmax classifier, a common multiclass classifier used in deep learning, follows the one-vs-all (or one-vs-rest) strategy, i.e., the output for a particular class can be interpreted as a probability value of the input belonging to that class (and conversely, the complement of the output, \((1-output)\), indicates the probability of the input not belonging to that class):
On the other hand, algorithms such as kNN, decision trees, etc. follow a one-vs-one strategy.
Multi-label classification
Multi-label classification is a type of classification task where each training example can have
more than one label
. The task of multi-label classification can be thought of as predicting the properties of a data-point that are not mutually exclusive, such as topics that are relevant for a document. A text might be about any of religion, politics, finance or education at the same time or none of these.
Examples
A multi-class problem has the assignment of instances to one of a finite, mutually-exclusive collection of classes. As in the example already given of crabs: male-blue, female-blue, male-orange, female-orange. Each of these is exclusive of the others and taken together they are comprehensive.
One form of a multi-label problem is to divide these into two labels, sex and color; where sex can be male or female, and color can be blue or orange. But note that this is a special case of the multi-label problem as every instance will get every label (i.e., every crab has both a sex and a color).
Multi-label problems also include other cases that allow for a variable number of labels to be assigned to each instance. For instance, an article in a newspaper or wire service may be assigned to the categories news, politics, sports, medicine, etc. One story about an important sporting event would get an assignment of the label “sports”; while another, involving political tensions that are revealed by a particular sporting event, might get both the labels “sports” and “politics”. For e.g., in the US, the results of the superbowl could be labeled both “sports” and “news” given the societal impact of the event.
Note that this form of labeling, with variable numbers of labels, can be recast into a form similar to the example with the crabs; except that every label is treated as label
X
or not label
X
. But not all methods require this recasting.
Key takeaways
:
The multi-class classification problem: one right answer, i.e., mutually exclusive outputs (for e.g., iris, numbers).
The multi-label classification problem: more than one right answer, i.e., non-exclusive outputs (for e.g., sugar test, eye test).
Graphical interpretation
To complement the other answers, here are some figures. Each row represents the expected output for one sample.
Multi-class classification
Recall that binary classification involves bucketing a sample into either of two categories. On the other hand, in the multi-class case, there are more than two classes in total.
One column = one class (one-hot encoding).
Multi-label classification
One column = one class,
In the multi-label case, one sample might be assigned more than one class.
Multilabel Multi-class classification
As a side note, nothing prevents you from having a multilabel multi-class classification problem, e.g.:
Multi-class vs. Multi-label Classification
References
Multi-class and Multi-label algorithms
What is the difference between Multi-label and Multi-class classification?
What is the difference between multiple outputs and Multi-label output?
Multi-class and Multi-label Classification, INFO-4604, Applied Machine Learning, University of Colorado Boulder
What is the difference between Multi-class and Multi-label Problem
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DistilledMulti-classvsMulti-labelClassification,
  title   = {Multi-class vs. Multi-label Classification},
  author  = {Chadha, Aman},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}