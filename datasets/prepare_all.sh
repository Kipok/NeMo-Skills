#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
python3 ${SCRIPT_DIR}/gsm8k/prepare.py --split_name train
python3 ${SCRIPT_DIR}/gsm8k/prepare.py --split_name validation
python3 ${SCRIPT_DIR}/gsm8k/prepare.py --split_name test
# train + validation - should be used for the final run only
python3 ${SCRIPT_DIR}/gsm8k/prepare.py --split_name train_full
python3 ${SCRIPT_DIR}/gsm-hard/prepare.py   # test set
python3 ${SCRIPT_DIR}/tabmwp/prepare.py     # test set
python3 ${SCRIPT_DIR}/mawps/prepare.py      # test set
python3 ${SCRIPT_DIR}/asdiv/prepare.py      # test set
python3 ${SCRIPT_DIR}/algebra222/prepare.py # test set
python3 ${SCRIPT_DIR}/svamp/prepare.py      # test set
python3 ${SCRIPT_DIR}/math/prepare.py --split_name train
python3 ${SCRIPT_DIR}/math/prepare.py --split_name validation
python3 ${SCRIPT_DIR}/math/prepare.py --split_name test
# train + validation - should be used for the final run only
python3 ${SCRIPT_DIR}/math/prepare.py --split_name train_full

# prepare datasets with masked solutions
python3 ${SCRIPT_DIR}/math-masked/prepare.py --split_name train
python3 ${SCRIPT_DIR}/math-masked/prepare.py --split_name validation
# train + validation - should be used for the final run only
python3 ${SCRIPT_DIR}/math-masked/prepare.py --split_name train_full
python3 ${SCRIPT_DIR}/gsm8k-masked/prepare.py --split_name train
python3 ${SCRIPT_DIR}/gsm8k-masked/prepare.py --split_name validation
# train + validation - should be used for the final run only
python3 ${SCRIPT_DIR}/gsm8k-masked/prepare.py --split_name train_full
