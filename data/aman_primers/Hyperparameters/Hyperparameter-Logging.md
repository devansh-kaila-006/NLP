# Hyperparameter Logging

**Source**: https://aman.ai/primers/ai/hyperparameter-logging
**Category**: Hyperparameters
**Word Count**: 1145

---

Introduction
Logging
Loading hyperparameters from a configuration file
Hyperparameter search
References
Citation
Introduction
In this section, we cover best practices to log, load hyperparameters and do random search.
Logging your outputs to a file is a general good practice in any project. An even more important good practice is to handle correctly the multiple hyperparameters that arise in any deep learning project.
A natural need thus arises to be able to log them and know the gamut of hyperparameters used in any past experiments.
Logging
A common problem when building a project is to forget about logging. In other words, as long as you write stuff in files and print things to the shell, people assume they’re going to be fine. A better practice is to write
everything
that you print to the terminal in a
log
file.
That’s why in
train.py
and
evaluate.py
we initialize a
logger
using the built-in
logging
package with:
#Set the logger to write the logs of the training in train.log
set_logger
(
os
.
path
.
join
(
args
.
model_dir
,
'train.log'
))
The
set_logger
function is defined in
utils.py
.
For instance, during training this line of code will create a
train.log
file in
experiments/base_model/
. You don’t have to worry too much about how we set it. Whenever you want to print somehting, use
logging.info
instead of the usual
print
:
logging
.
info
(
"It will be printed both to the Terminal and written in the .log file"
)
That way, you’ll be able to both see it in the Terminal and remember it in the future when you’ll need to read the
train.log
file.
Loading hyperparameters from a configuration file
You’ll quickly realize when doing a final project or any research project that you’ll need a way to specify some parameters to your model. You have different sorts of hyperparameters (not all of them are necessary):
hyperparameters for the model: number of layers, number of neurons per layer, activation functions, dropout rate…
hyperparameters for the training: number of epochs, learning rate, …
dataset choices: size of the dataset, size of the vocabulary for text, …
checkpoints: when to save the model, when to log to plot the loss, …
There are multiple ways to load the hyperparameters:
Use the
argparse
module as we do to specify the
data_dir
:
parser
.
add_argument
(
'--data_dir'
,
default
=
'data/'
,
help
=
"Directory containing the dataset"
)
When experimenting, you need to try multiples combinations of hyperparameters. This quickly becomes unmanageable because you cannot keep track of the hyperparameters you are testing. Plus, how do you even keep track of the parameters if you want to go back to a previous experiment?
Hard-code the values of your hyperparameters in a new
params.py
file and import at the beginning of your
train.py
file for instance, get these hyperparameters. Again, you’ll need to find a way to save your config, and this is not very clean.
Write all your parameters in a file (we used
.json
but could be anything else) and store this file in the directory containing your experiment. If you need to go back to your experiment later, you can quickly review which hyperparameters yielded the performance etc.
We chose to take this third approach in our code. We define a class
Params
in
utils.py
. Note that to be in accordance with the deep learning programming frameworks we use, we are refering to hyperparameters as
params
in the code.
Loading the hyperparameters is as simple as writing
params
=
Params
(
"experiments/base_model/params.json"
)
and if your params.json file looks like
{
"model_version"
:
"baseline"
,
"learning_rate"
:
1e-3
,
"batch_size"
:
32
,
"num_epochs"
:
10
}
you’ll be able to access the different entries with
params
.
model_version
In your code, once your params object is initialized, you can update it with another
.json
file with the
params.update("other_params.json")
method.
Later, in your code, for example when you define your model, you can thus do something like
if
params
.
model_version
==
"baseline"
:
logits
=
build_model_baseline
(
inputs
,
params
)
elif
params
.
model_version
==
"simple_convolutions"
:
logits
=
bulid_model_simple_convolutions
(
inputs
,
params
)
which will be quite handy to have different functions and behaviors depending on a set of hyperparameters!
Hyperparameter search
An important part of any machine learning project is hyperparameter tuning, please refer to the Coursera Deep Learning Specialization (
#2
and
#3
) for detailed information. In other words, you want to see how your model performs on the development set on different sets of hyperparameters. There are basically two ways to implement this:
Have a python loop over the different set of hyperparameters and at each iteration of the loop, run the
train_and_evaluate(model_spec, params, ...)
function, like
for
lr
in
[
0.1
,
0.01
,
0.001
]:
params
.
learning_rate
=
lr
train_and_evaluate
(
model_spec
,
params
,
...)
Have a more general script that will create a sub-folder for each set of hyperparameters and launch a training job using the
python train.py
command. While there is not much difference in the simplest setting, some more advanced clusters have some job managers and instead of running multiple
python train.py
, they instead do something like
job-manager-submit train.py
which will run the jobs concurrently, making the hyperparameter tuning much faster!
for
lr
in
[
0.1
,
0.01
,
0.001
]:
params
.
learning_rate
=
lr
#Create new experiment directory and save the relevant params.json
subfolder
=
create_subfolder
(
"lr_{}"
.
format
(
lr
))
export_params_to_json
(
params
,
subfolder
)
#Launch a training in this directory -- it will call `train.py`
lauch_training_job
(
model_dir
=
subfolder
,
...)
This is what the
search_hyperparams.py
file does. It is basically a python script that runs other python scripts. Once all the sub-jobs have ended, you’ll have the results of each experiment in a
metrics_eval_best_weights.json
file for each experiment directory.
learning_rate
/
hyperparams
.
json
learning_rate_0
.
1
/
hyperparams
.
json
metrics_eval_best_weights
.
json
learning_rate_0
.
01
/
hyperparams
.
json
metrics_eval_best_weights
.
json
and by running
python synthesize_results.py --model_dir experiments/learning_rate
you’ll be able to gather the different metrics achieved for the different sets of hyperparameters!
From one experiment to another, it is very important to test hyperparameters one at a time. Comparing the dev-set performance of two models “A” and “B” which have a totally different set of hyperparameters will probably lead to wrong decisions. You need to vary only ONE hyperparameter (let’s say the learning rate) when comparing models “A” and “B”. Then, you can see the impact of this change on the dev-set performance.
References
Structuring Machine Learning Projects on Coursera
CS230 code examples
Citation
If you found our work useful, please cite it as:
@article{Chadha2020DistilledHyperparameterTuning,
  title   = {Splitting Datasets},
  author  = {Chadha, Aman},
  journal = {Distilled AI},
  year    = {2020},
  note    = {\url{https://aman.ai}}
}