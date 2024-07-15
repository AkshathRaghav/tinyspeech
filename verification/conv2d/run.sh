module load gcc/9.3.0 
module load cuda/12.1.1 cudnn/cuda-12.1_8.9 

cd ../../
source ./.venv/bin/activate
cd verification/conv2d 

python test.py 
gcc test.c -o test
./test