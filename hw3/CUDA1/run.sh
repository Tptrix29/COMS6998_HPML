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
MATRIX_SIZE=(256 512 1024)
for s in ${MATRIX_SIZE[@]}
do
echo "Running with MATRIX_SIZE=$s"
    ./matmult00 $(expr $s / 16)
    ./matmult01 $(expr $s / 32)
done