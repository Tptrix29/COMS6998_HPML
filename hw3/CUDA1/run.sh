make clean
make

# vecadd
V_PER_THREAD=(500 1000 2000)
for v in ${V_PER_THREAD[@]}
do
    echo "Running with V_PER_THREAD=$v"
    ./vecadd00 $v
    ./vecadd01 $v
done

# matmult
