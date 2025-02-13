echo "Bechmarking dp1:"
echo "N = 1000000, R = 1000:"
./dp1 1000000 1000
echo "N = 300000000, R = 20:"
./dp1 300000000 20
echo -e "\n"

echo "Bechmarking dp2:"
echo "N = 1000000, R = 1000:"
./dp2 1000000 1000
echo "N = 300000000, R = 20:"
./dp2 300000000 20
echo -e "\n"


echo "Bechmarking dp3:"
echo "N = 1000000, R = 1000:"
./dp3 1000000 1000
echo "N = 300000000, R = 20:"
./dp3 300000000 20
echo -e "\n"

echo "Bechmarking dp4:"
echo "N = 1000000, R = 1000:"
python3 dp4.py -N 1000000 -R 1000
echo "N = 300000000, R = 20:"
python3 dp4.py -N 300000000 -R 20
echo -e "\n"

echo "Bechmarking dp5:"
echo "N = 1000000, R = 1000:"
python3 dp5.py -N 1000000 -R 1000
echo "N = 300000000, R = 20:"
python3 dp5.py -N 300000000 -R 20