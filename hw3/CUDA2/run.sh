make clean
make
echo -e "\n"

K_ARRAY=(1 5 10 50 100)
for k in ${K_ARRAY[@]}
do
    echo "Running with K=$k"
    echo "CPU-based:"
    ./q1 $k
    echo "Non-unified memory: "
    ./q2 $k
    echo "Unified memory: "
    ./q3 $k
    echo -e "\n"
done