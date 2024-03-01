Before using this tool, you first need to prepare dataset and install requirements:
```
    pip install -r visualization/requirements.txt
    ./datasets/prepare_all.sh
```

By default 'math' dataset will be used. You can change it in config.yaml file (dataset: str = "math"), in visualization/settings/visualization_config.yaml or through command line

Command to launch the program (all parameters are optional):
```
    python3 visualization/__main__.py \
    ssh_key=<path_to_ssh> \
    ssh_server=<server> \
    hostname=<host> \
    dataset=<dataset_name>
```

For instance:
```
    python3 visualization/__main__.py \
    ssh_key=~/.ssh/id_rsa.pub \
    ssh_server=username@server_name \
    hostname=10.180.11.36 \
    dataset=gsm8k
```

To set the path to your results files you should change visualization/settings/visualization_config.yaml file:

```
prediction_jsonl_files:
    model1: /some_path/model1/output-greedy.jsonl
    model2: /some_path/model2/output-rs*.jsonl
```