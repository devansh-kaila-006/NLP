# Learning Paradigms

**Source**: https://aman.ai/primers/ai/learning-paradigms
**Category**: Data Foundations
**Word Count**: 833

---

Overview
Types of learning
Self-supervised learning (SSL)
SSL Tasks
The Cloze Task / Masked Language Modeling
Next Token Prediction
Visual Summary
Learning Process
Transformers
Pretraining
Alignment
Visual Summary
Further Reading
Citation
Overview
Self-supervised learning is a key advancement that revolutionized natural language processing and generative AI. Here’s how it works and two examples of how it is used to train language models.
Self-supervised learning is a key advancement in deep learning that is used across a variety of domains. Put simply, the idea behind self-supervised learning is to train a model over raw/unlabeled data by making out and predicting portions of this data. This way, the ground truth “labels” that we learn to predict are present in the data itself.
Types of learning
Machine learning models can be trained in a variety of ways. For example, supervised learning trains a machine learning model over pairs of input data and output labels (usually annotated manually by humans). The model learns to predict these output labels by supervising the model! On the other hand, unsupervised learning uses no output labels and discovers inherent trends within the input data itself (e.g., by forming clusters).
“Self-supervised learning obtains supervisory signals from the data itself, often leveraging the underlying structure in the data. The general technique of self-supervised learning is to predict any unobserved or hidden part (or property) of the input from any observed or unhidden part of the input.” - from
Self-supervised learning: The dark matter of intelligence
.
Self-supervised learning (SSL)
Self-supervised learning (SSL) lies between supervised and unsupervised learning. Namely, we train the model over pairs of input data and output labels. However, no manual annotation from humans is required to obtain output labels within our training data—the labels are naturally present in the raw data itself! To understand this better, let’s take a look at a few commonly-used self-supervised learning objectives.
SSL Tasks
The Cloze Task / Masked Language Modeling
The Cloze task is more commonly referred to as the masked language modeling (MLM) objective. Here, the language model takes a sequence of textual tokens (i.e., a sentence) as input. To train the model, we mask out (i.e., set to a special “mask” token) ~10% of tokens in the input and train the model to predict these masked tokens. Using this approach, we can train a language model over an unlabeled textual corpus, as the “labels” that we predict are just tokens that are already present in the text itself. This objective is used to pretrain language models like BERT and T5.
Next Token Prediction
Next token prediction is the workhorse of modern generative language models like ChatGPT and PaLM. After downloading a large amount of raw textual data from the internet, we can repeatedly i) sample a sequence of text and ii) train the language model to predict the next token given preceding tokens as input. This happens in parallel for all tokens in the sequence. Again, all the “labels” that we learn to predict are already present in the raw textual data. Pretraining (and finetuning) via next token prediction is universal used by all generative language models.
Visual Summary
Learning Process
Here’s a simple, three-part framework to understand the learning process of generative language models.
Transformer architecture: the neural network architecture used by LLMs.
Language model pretraining: the (initial) training process used by LLMs.
The alignment process: how we teach LLMs to behave to our liking.
Transformers
Most recent generative language models are based upon the transformer architecture. Although the transformer was originally proposed with two modules (i.e., an encoder and a decoder), generative LLMs use a decoder-only variant of this architecture. This architecture takes as input a sequence of tokens (i.e., words or subwords) that have been embedded into a corresponding vector representation and transforms them via masked self-attention and feed-forward transformations.
Pretraining
The most commonly-used objective for pretraining is next token prediction, also known as the standard language modeling objective. Interestingly, this objective—despite being quite simple to understand—is the core of all generative language models. To pretrain a generative language model, we curate a large corpus of raw text and iteratively perform the following steps:
Sample a sequence of raw text from the dataset.
Pass this textual sequence through the decoder-only transformer.
Train the model to accurately predict the next token at each position within the sequence.
Alignment
After pretraining, the LLM can accurately perform next token prediction, but its output is oftentimes repetitive and uninteresting. The alignment process teaches a language model how to generate text that aligns with the desires of a human user. To align a language model, we define a set of alignment criteria (e.g., helpful and harmless) and finetune the model (using SFT and RLHF) based on these criteria.
Visual Summary
Further Reading
Explaining ChatGPT to Anyone in <20 Minutes
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DistilledLossFunctions,
  title   = {Loss Functions},
  author  = {Chadha, Aman and Jain, Vinija},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://vinija.ai}}
}