This is a tool for data analysis. It has two pages: "Inference" and "Analyze".
## Inference page
This page allows for the analysis of model answers depending on different parameters we set up. It has three modes: "Chat", "Run one sample", and "Run whole dataset".

**Chat** mode allows for a conversation with the model. It has the minimum parameters to set up.

**Run one sample** mode allows you to send a single question to the model. It can be a question from the dataset (parameters `data_file` or `dataset` and `split_name` determine the dataset) or your custom question. The answer is also validated by comparing it with the `expected_answer` field.

**Run whole dataset** mode allows you to launch the model with chosen parameters on the whole dataset. It will save results in `visualization/results/output-greedy.jsonl` and `visualization/results/metrics-greedy.jsonl`. If the "use random seed range" flag is turned on, the dataset will be launched for random seeds in the range from `start_random_seed` to `end_random_seed`. All results can then be analyzed on the "Analyze" page (you do not need to relaunch the program), and the parameters you used to launch the dataset can be found in the `visualization/results/parameters.jsonl` file and on the "Analyze" page as well.

## Analyze page
To use the Analyze page, you need to specify paths to the datasets you want to analyze (if you did not get the data through the "Inference" page). You can pass parameters via command line  `++visualization_params.model_prediction.model1='/some_path/model1/output-greedy.jsonl'` or modify [visualization_config.yaml](/visualization/settings/visualization_config.yaml)
```yaml
visualization_params:
  prediction_jsonl_files:
    model1: /some_path/model1/output-greedy.jsonl
    model2: /some_path/model2/output-rs*.jsonl
```
`model1` and `model2` are the names of the datasets we want to analyze. All files satisfying the given pattern will be taken for analysis.
On the page itself, you can sort, filter, and compare models. You can also add labels to the data and save your modified, filtered, and sorted dataset by specifying `save_dataset_path`.

## Getting Started
Before using this tool, you first need to do instructions in [prerequisites.md](/docs/prerequisites.md), download datasets and install requirements:
```
    pip install -r visualization/requirements.txt
    ./datasets/prepare_all.sh
```

You can change parameters in the [config.yaml](/visualization/settings/config.yaml) file, in [visualization_config.yaml](/visualization/settings/visualization_config.yaml), or through the command line.
Command to launch the program (all parameters are optional):
```
python visualization/data_explorer.py \
++server.ssh_key_path=<path_to_ssh> \
++server.ssh_server=<server> \
++server.host=<host>
```
For the "Inference" page, you need to specify `ssh_key`, `ssh_server`, and `host` and launch the server with the model (see `docs/inference.md`).
