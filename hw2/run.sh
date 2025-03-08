export export PATH=$PATH:/sbin

# C2
echo "C2 Start..."
python3 main.py --cuda --epochs 5 --batch_size 128 --random_seed 42 --worker 2 > C2.log 2>&1
echo "C2 Done"

# C3
rm -f C3.log
echo "C3 Start..."
workers=(0 4 8 12 16 20)
for w in ${workers[@]}; do
    echo "Benchmarking with $w workers..." >> C3.log
    python3 main.py --cuda --epochs 5 --batch_size 128 --random_seed 42 --worker $w >> C3.log 2>&1
    echo "Done" >> C3.log
    echo >> C3.log
done
echo "C3 Done"

# C4
echo "C4 Start..."
python3 main.py --cuda --epochs 5 --batch_size 128 --random_seed 42 --worker 1 > C4.log 2>&1
echo "C4 Done"

# C5
echo "C5 Start..."
python3 main.py --epochs 5 --batch_size 128 --random_seed 42 --worker 4 > C5.log 2>&1
echo "C5 Done" 

# C6
rm -f C6.log
echo "C6 Start..."
optimizers=(sgd nestrov adagrad adadelta adam)
for optim in ${optimizers[@]}; do
    echo "Benchmarking with $optim optimizer..." >> C6.log
    python3 main.py --cuda --epochs 5 --batch_size 128 --random_seed 42 --worker 4 --optim $optim >> C6.log 2>&1
    echo "Done" >> C6.log
    echo >> C6.log
done
echo "C6 Done"

# C7
echo "C7 Start..."
python3 main.py --cuda --epochs 5 --batch_size 128 --random_seed 42 --worker 4 --disable_batch_norm > C7.log 2>&1
echo "C7 Done"

# C8
rm -f C8.log
echo "C8 Start..."
compilation_modes=(eager default reduce autotune)
for mode in ${compilation_modes[@]}; do
    echo "Benchmarking with $mode compilation mode..." >> C8.log
    python3 main.py --cuda --epochs 10 --batch_size 128 --random_seed 42 --worker 4 --compile $mode --droplast >> C8.log 2>&1
    echo "Done" >> C8.log
    echo >> C8.log
done
echo "C8 Done"

