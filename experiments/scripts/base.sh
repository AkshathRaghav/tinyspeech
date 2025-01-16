cd /depot/euge/data/araviki/vsdsquadronmini/
module load gcc/9.3.0 
module load cuda/12.1.1 cudnn/cuda-12.1_8.9 
source ./.venv/bin/activate.csh

python3 ./training/train.py --config ./experiments/DiffTest.yaml



python3 ./training/train.py --config ./experiments/tinyspeechz_google_speech.yaml
    