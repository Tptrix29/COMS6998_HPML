# COMS6998_HPML
Homework for High Performance Machine Learning course at Columbia University

## Setup
1. GCP Environment
2. Python Virtual Environment
    ```shell
    python3 -m venv env
    source env/bin/activate
    pip3 install -r requirements.txt
    ```
## Contents
- Homework 1: Performance Comparison of Vector Dot Product
    ```shell
    cd hw1
    make  # compile C code
    bash benchmark.sh  # run benchmarks
    ```