#!/bin/bash
#SBATCH -J vqbet_12_a100 #job name
#SBATCH --time=01-00:00:00 #requested time (DD-HH:MM:SS)
#SBATCH -p gpu #running on "mpi" partition/queue
#SBATCH --gres=gpu:a100:1 #requesting 1 GPU
#SBATCH --constraint="a100-80G"
#SBATCH -N 1 #1 nodes
#SBATCH -n 8 #2 tasks total (default 1 CPU core per task) = # of cores
#SBATCH --mem=64g #RAM total
#SBATCH --output=MyJob.%j.%N.out #saving standard output to file, %j=JOBID, %N=NodeName
#SBATCH --error=MyJob.%j.%N.err #saving standard error to file, %j=JOBID, %N=NodeName
#SBATCH --mail-type=ALL #email optitions
#SBATCH --mail-user=james.staley625703@tufts.edu 


#[commands_you_would_like_to_exe_on_the_compute_nodes] 
# for example, running a python script
# 1st, load the modulemodule load anaconda/2021.05
# run pythonpython myscript.py #make sure myscript.py exists in the current directory or provide thefull path to script

module load anaconda/2021.11
module load cuda/12.2
export DATA_DIR='/cluster/tufts/shortlab/jstale02/lerobot/local/'
export WANDB_DATA_DIR=/cluster/tufts/shortlab/jstale02
export WANDB_CACHE_DIR=/cluster/tufts/shortlab/jstale02
sleep 5
source activate three_ten
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/7' env.image_size=64 hydra.job.name='imi7'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/8' env.image_size=64 hydra.job.name='imi8'

python lerobot/scripts/train.py policy=vqbet env=pusht dataset_repo_id='pusht/12_96x96' hydra.job.name='imi12_96' seed=5632 wandb.project='vqbet_pusht'
python lerobot/scripts/train.py policy=vqbet env=pusht dataset_repo_id='pusht/A12_96x96' hydra.job.name='Aimi12_96' seed=5632 wandb.project='vqbet_pusht'