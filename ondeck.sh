# AI Dems
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A11_96x96' hydra.job.name='Aimi11_96' seed=7
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/11_96x96' hydra.job.name='imi11_96' seed=7

# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A12_96x96' hydra.job.name='Aimi12_96' seed=5632
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/12_96x96' hydra.job.name='imi12_96' seed=5632

# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A13_96x96' hydra.job.name='Aimi13_96' seed=424213
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/13_96x96' hydra.job.name='imi13_96' seed=423213
# DATA_DIR='./local' python lerobot/scripts/train.py policy=vqbet env=pusht dataset_repo_id='pusht/14_96x96' hydra.job.name='imi14_96' seed=6213 wandb.project='vqbet_pusht'
# DATA_DIR='./local' python lerobot/scripts/train.py policy=vqbet env=pusht dataset_repo_id='pusht/A14_96x96' hydra.job.name='Aimi14_96' seed=6213 wandb.project='vqbet_pusht'

# sleep 5
# LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_20' hydra.job.name='20' seed=95113 wandb.project='diffusion_pusht'
# sleep 5
# LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_21' hydra.job.name='21' seed=3 wandb.project='diffusion_pusht'
# sleep 5
# LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_22' hydra.job.name='22' seed=35 wandb.project='diffusion_pusht'
# sleep 5
# LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_24' hydra.job.name='24' seed=52313 wandb.project='diffusion_pusht'
# sleep 5
# DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='20out' hydra.job.name='20out' seed=52873 wandb.project='diffusion_pusht'
# DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='21out' hydra.job.name='21out' seed=554433 wandb.project='diffusion_pusht'
# DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='22out' hydra.job.name='22out' seed=22353 wandb.project='diffusion_pusht'
# DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='23out' hydra.job.name='23out' seed=183953 wandb.project='diffusion_pusht'
# DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='23out' hydra.job.name='23out' seed=183953 wandb.project='diffusion_pusht'

#LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_A20' hydra.job.name='A20' seed=51555 wandb.project='diffusion_pusht'
#sleep 5
# LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_A21' hydra.job.name='A21' seed=784 wandb.project='diffusion_pusht'
#sleep 5
# LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_A22' hydra.job.name='A22' seed=425 wandb.project='diffusion_pusht'
# sleep 5
# LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_A23' hydra.job.name='A23' seed=6601 wandb.project='diffusion_pusht'
# sleep 5

# LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_A24' hydra.job.name='A24' seed=112 wandb.project='diffusion_pusht'
# sleep 5
# LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_24' hydra.job.name='24' seed=94909 wandb.project='diffusion_pusht'
# sleep 5
# LEROBOT_HOME='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='20out' hydra.job.name='20out' seed=52873 wandb.project='diffusion_pusht'
# sleep 5
LEROBOT_HOME='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='21out' hydra.job.name='21out' seed=554433 wandb.project='diffusion_pusht'
sleep 5
LEROBOT_HOME='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='22out' hydra.job.name='22out' seed=22353 wandb.project='diffusion_pusht'
sleep 5
LEROBOT_HOME='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='23out' hydra.job.name='23out' seed=183953 wandb.project='diffusion_pusht'
sleep 5
LEROBOT_HOME='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='24out' hydra.job.name='24out' seed=85319 wandb.project='diffusion_pusht'

## Waiting for the 24 model to train before the 24out demonstrations can be generated and this can run
# python lerobot/scripts/evalv2.py -p ./local/models/24 eval.n_episodes=100 eval.batch_size=100
# DATA_DIR='./local/models' python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='23out' hydra.job.name='23out' seed=183953 wandb.project='diffusion_pusht'
