Before using this tool, you first need to prepare dataset and install requirements.txt
```
    ./datasets/prepare_all.sh
```
(see readme in root folder for more details)

By default 'math' dataset will be used. You can change it in config.yaml file (dataset: str = "math"), in visualization/settings/base/models_predictions.yaml or through command line

Command to launch the program (all parameters are optional):
```
    python3 visualization/__main__.py \
    ssh_key={path_to_ssh} \
    ssh_server={server} \
    hostname={host} \
    dataset={dataset_name}
```

For instance:
```
    python3 visualization/__main__.py \
    ssh_key=~/.ssh/id_rsa.pub \
    ssh_server=igitman@draco-rno-login.nvidia.com \
    hostname=10.180.11.36 \
    dataset=gsm8k
```

To set the path to your results files you should change visualization/settings/base/models_predictions.yaml file

```
prediction_jsonl_files:
  name: results/math_with/output-rs*
  name1: results/math_without/output-rs*
```