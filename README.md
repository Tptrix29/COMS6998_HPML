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
    Prerequisites: Setup MKL library for Intel CPU at GCP: 
    ```shell
    wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null

    echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list

    sudo apt update

    sudo apt install intel-basekit
    sudo apt install git
    sudo apt-get install nohup make
    sudo apt-get install python3-venv

    echo . /opt/intel/oneapi/setvars.sh >> .bashrc
    source .bashrc
    ```
    Run benchmarks:
    ```shell
    cd hw1
    make  # compile C code
    bash benchmark.sh  # run benchmarks
    ```
- Homework 2: Performance Benchmark for ResNet18
