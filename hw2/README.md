## COMS6998: Assignment 2

### Files
- `README.md`: Instructions for the assignment
- `requirements.txt`: Python dependencies
- `model.py`: ResNet model definition
- `main.py`: Main script to train and evaluate the model
- `run.sh`: Script for all the experiments
- `extra.sh`: Script for extra credit
- `HW2.ipynb`: Notebook for analysis and plots


### Usage
```shell
# Install dependencies
pip install -r requirements.txt
# Create virtual environment
python3 -m venv env
source env/bin/activate
# Env setup
sudo apt-get update
sudo apt-get install libc-bin
export PATH=$PATH:/sbin
# Show help
python main.py -h
# Benchmark
bash run.sh
# Next: Analyze the results in HW2.ipynb

# Extra credit
bash extra.sh
# Visualize via TensorBoard
tensorboard --logdir=log
```
