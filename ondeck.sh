# export DATA_DIR='./local/'

# source venv/bin/activate

# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A4' env.image_size=64 hydra.job.name='Aimi4'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A5' env.image_size=64 hydra.job.name='Aimi5'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A6' env.image_size=64 hydra.job.name='Aimi6'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A7' env.image_size=64 hydra.job.name='Aimi7'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/4' env.image_size=64 hydra.job.name='imi4'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/5' env.image_size=64 hydra.job.name='imi5'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/6' env.image_size=64 hydra.job.name='imi6'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/11' env.image_size=64 hydra.job.name='imi11'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/12' env.image_size=64 hydra.job.name='imi12'
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/13' env.image_size=64 hydra.job.name='imi13' training.offline_steps=300000
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/14' env.image_size=64 hydra.job.name='imi14' training.offline_steps=300000


# AI Dems
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A11_96x96' hydra.job.name='Aimi11_96' seed=7
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/11_96x96' hydra.job.name='imi11_96' seed=7

# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A12_96x96' hydra.job.name='Aimi12_96' seed=5632
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/12_96x96' hydra.job.name='imi12_96' seed=5632

# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/A13_96x96' hydra.job.name='Aimi13_96' seed=423213
# python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pusht/13_96x96' hydra.job.name='imi13_96' seed=423213
# DATA_DIR='./local' python lerobot/scripts/train.py policy=vqbet env=pusht dataset_repo_id='pusht/14_96x96' hydra.job.name='imi14_96' seed=6213 wandb.project='vqbet_pusht'
# DATA_DIR='./local' python lerobot/scripts/train.py policy=vqbet env=pusht dataset_repo_id='pusht/A14_96x96' hydra.job.name='Aimi14_96' seed=6213 wandb.project='vqbet_pusht'
sleep 5
LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_20' hydra.job.name='20' seed=95113 wandb.project='diffusion_pusht'
sleep 5
LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_20' hydra.job.name='21' seed=3 wandb.project='diffusion_pusht'
sleep 5
LEROBOT_HOME=./local/ python lerobot/scripts/train.py policy=diffusion env=pusht dataset_repo_id='pushtv2_20' hydra.job.name='23' seed=513 wandb.project='diffusion_pusht'
sleep 5
