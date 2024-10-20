# Greedy Ensemble Selection

---
This is the implementation for our study on greedy ensemble selection for top-n ranking prediction. 

## Installation

---
This project was tested with Python 3.9 on Windows, Mac, and Linux.  
You can install the required packages using the `requirements.txt` file.  

## Usage

---
This program has one main entry points which requires you to set up an experiment configuration file.  
The file `experiment_template.json` serves as an example configuration.  
Make a copy of this file, configure your experiment, and replace `template` in the file name with your desired
experiment name.

The list below details all the configuration options inside the configuration file:

| Option                          | Description                                                                     |
|---------------------------------|---------------------------------------------------------------------------------|
| `DATA_SET_NAMES`                | Comma-separated list of data sets.                                              |
| `NUM_FOLDS`                     | The number of folds for cross-validation.                                       |
| `RECOMMENDERS`                  | Comma-separated list of recommenders.                                           |
| `NUM_BATCHES`                   | Number of user batches that are predicted for. Increases parallelization.       |
| `TOPN_SCORE`                    | The top number of items to select from.                                         |
| `METRICS`                       | The metrics to evaluate for.                                                    |
| `RUN_MODE`                      | Mode of optimization.                                                           |

### Supported Data Sets

This framework natively supports 25 data sets.  
New data sets can be added by implementing the necessary load function.  
In theory, support for any data set with user-item interactions can be used.  
To use any supported data set you need to download it and place it in the `data` folder inside the project root.  
Specifically, each data set needs to be placed in a folder with a specific name such that the loading routing can find
it.  
Inside that folder, the raw data set has to be placed in a folder called `original`.  
Example: To use the MovieLens 100k data set, download it, and place the necessary data file in the following
location: `data/ml-100k/original/u.data`.

The following table lists the supported data sets with their precise folder name, download link, note, feedback type and
domain:

| Data Set Name       | Download                                                                             | Notes                                                                                          | Feedback Type | Domain    |
|---------------------|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|---------------|-----------|
| `ciaodvd`           | https://guoguibing.github.io/librec/datasets.html                                    | Only `movie-ratings.txt` required.                                                             | Explicit      | Movies    |
| `citeulike-a`       | https://github.com/js05212/citeulike-a                                               | Only `users.dat` required.                                                                     | Implicit      | Articles  |
| `hetrec-lastfm`     | https://grouplens.org/datasets/hetrec-2011/                                          | Only `user_artists.dat` required.                                                              | Implicit      | Music     |
| `ml-1m`             | https://grouplens.org/datasets/movielens/                                            | Only `ratings.dat` required.                                                                   | Explicit      | Movies    |
| `ml-100k`           | https://grouplens.org/datasets/movielens/                                            | Only `u.data` required.                                                                        | Explicit      | Movies    |

### IMPORTANT: Execution order

Note that full execution of experiments is a XXX stage process.  
The execution has to happen in sequential order, e.g., stage 2 cannot be executed before stage 1.  
The stages are as follows:
<ol start="0">
    <li>Data pruning. The data is read from the original file(s), cleaned of duplicates, pruned, re-mapped, and saved in a homogeneous format.</li>
    <li>Data splitting. The data is split into folds for cross-validation. Splitting is performed per user to avoid user cold start.</li>
    <li>Recommender fitting. The recommenders are fitted on the training data.</li>
    <li>Recommender predicting. The fitted recommenders predict a ranked list for each user.</li>
	<li>Aggregate predictions. The prediction process is parallelized. Therefore, all predictions need to be aggregated.</li>
	<li>Create prediction matrix. Combine all model predictions into one matrix.</li>
	<li>Execute Ensembling. Run greedy ensemble selection.</li>
</ol>

###  Local execution

Local execution requires a Python environment with the required packages.
The entry point is `local_executor.py`.  
The configuration is controlled via `select_experiment.py`.  
Open `select_experiment.py`, make and save changes, then run `local_executor.py`.  
Example: `python local_executor.py`.
