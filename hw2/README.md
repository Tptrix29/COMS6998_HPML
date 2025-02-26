## COMS6998: Assignment 2

### Files
- `README.md`: Instructions for the assignment
- `model.py`: ResNet model definition
- `main.py`: Main script to train and evaluate the model

- `C3.sh`: Script for C3 questison: find out optimal worker number

### Usage
```shell
# Install dependencies
pip install -r requirements.txt
# Show help
python main.py -h
# Train and evaluate the model
python main.py --input_dir <path_to_input_dir> --optim <optimizer> --epochs <num_epochs> --worker <num_workers>
# Example 1: Train and evaluate the model with SGD optimizer on CPU
python main.py --input_dir ./data --optim sgd --epochs 5 --worker 2
# Example 2: Train and evaluate the model with SGD optimizer on GPU
python main.py --input_dir ./data --optim sgd --epochs 5 --worker 2 --cuda 
```
