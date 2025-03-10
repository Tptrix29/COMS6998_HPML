rm -f profile.log
# C2
echo "Profiling for C2..."
python3 main.py --cuda --epochs 5 --batch_size 128 --random_seed 42 --worker 2 --profile >> profile.log 2>&1
echo "C2 Done"

# C3
echo "Profiling for C3..."
workers=(0 4 8)
for w in ${workers[@]}; do
    echo "Benchmarking with $w workers..." >> profile.log
    python3 main.py --cuda --epochs 5 --batch_size 128 --random_seed 42 --worker $w --profile >> profile.log 2>&1
    echo "Done" >> profile.log
    echo
done
echo "C3 Done"

# C4
echo "Profiling for C4..."
python3 main.py --cuda --epochs 5 --batch_size 128 --random_seed 42 --worker 1 --profile >> profile.log 2>&1
echo "C4 Done"

# C5
echo "Profiling for C5..."
python3 main.py --epochs 5 --batch_size 128 --random_seed 42 --worker 4 --profile >> profile.log 2>&1
echo "C5 Done"