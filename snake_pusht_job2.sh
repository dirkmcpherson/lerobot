#!/bin/bash
#SBATCH -J snake_diffusion_a100 #job name
#SBATCH --time=03-00:00:00 #requested time (DD-HH:MM:SS)
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

# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A14_96x96' hydra.job.name='Aimi14_96' seed=6213

# python lerobot/scripts/train.py policy=vqbet env=pusht dataset_repo_id='pusht/13_96x96' hydra.job.name='imi13_96' seed=423213 wandb.project='vqbet_pusht'
# python lerobot/scripts/train.py policy=vqbet env=pusht dataset_repo_id='pusht/A13_96x96' hydra.job.name='Aimi13_96' seed=423213 wandb.project='vqbet_pusht'
# DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='12out' hydra.job.name='12out' seed=43 wandb.project='diffusion_pusht'
# DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='13out' hydra.job.name='13out' seed=13 wandb.project='diffusion_pusht'
DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='14out' hydra.job.name='14out' seed=21 wandb.project='diffusion_pusht'
DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='11out' hydra.job.name='11out' seed=4213 wandb.project='diffusion_pusht'