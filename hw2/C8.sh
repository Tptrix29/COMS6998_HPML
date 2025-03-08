python3 main.py --cuda --epochs 10 --batch_size 128 --random_seed 42 --worker 4 --compile eager
python3 main.py --cuda --epochs 10 --batch_size 128 --random_seed 42 --worker 4 --compile default
python3 main.py --cuda --epochs 10 --batch_size 128 --random_seed 42 --worker 4 --compile reduce
python3 main.py --cuda --epochs 10 --batch_size 128 --random_seed 42 --worker 4 --compile autotune
