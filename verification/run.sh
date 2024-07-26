module load gcc/9.3.0 
module load cuda/12.1.1 cudnn/cuda-12.1_8.9 

cd ../../
source ./.venv/bin/activate.csh
cd verification/conv2d 

python test.py --scale 0.25 --bpw "8bit" && gcc test.c -o conv2d -lm && valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out1.txt ./conv2d