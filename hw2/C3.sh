# Find out the optimal #workers
workers=(0 4 8 12 16 20)
for w in ${workers[@]}; do
    echo "Benchmarking with $w workers..."
    python3 main.py --cuda --epochs 5 --batch_size 128 --random_seed 42 --worker $w
    echo "Done"
    echo
done

