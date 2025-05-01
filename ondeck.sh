# LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=js/ros_010 policy=diffusion_ros_real env=ros_real hydra.run.dir=outputs/train/dp_ros_010 hydra.job.name=dp_ros_010 device=cuda wandb.enable=false training.save_freq=20000

# LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=js/ros_011 policy=diffusion_ros_real env=ros_real hydra.run.dir=outputs/train/dp_ros_011 hydra.job.name=dp_ros_011 device=cuda wandb.enable=false training.save_freq=20000

# LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=js/ros_009 policy=diffusion_ros_real env=ros_real hydra.run.dir=outputs/train/dp_ros_009 hydra.job.name=dp_ros_009 device=cuda wandb.enable=false training.save_freq=20000


LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=local/ros_54_450 policy=diffusion_ros_real env=ros_real hydra.run.dir=outputs/train/dp_ros_54_450 hydra.job.name=dp_ros_54_450 device=cuda wandb.enable=false training.save_freq=10000 && LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=local/ros_54_450_BOT policy=diffusion_ros_realB env=ros_real hydra.run.dir=outputs/train/dp_ros_54_450B hydra.job.name=dp_ros_54_450B device=cuda wandb.enable=false training.save_freq=10000
LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=local/genesis_0_None policy=diffusion_genesis env=genesis hydra.run.dir=outputs/train/dp_diffusion hydra.job.name=dp_diffusion device=cuda wandb.enable=false training.save_freq=20000 training.eval_freq=-1
python lerobot/scripts/eval.py     -p outputs/train/dp_diffusion/checkpoints/last/pretrained_model/     eval.n_episodes=1     eval.batch_size=1


LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=local/genesis_0_None policy=diffusion_genesis env=genesis hydra.run.dir=outputs/train/dp_diffusion_ntt20 hydra.job.name=dp_diffusion_ntt20 device=cuda wandb.enable=false training.save_freq=20000 training.eval_freq=-1
python lerobot/scripts/eval.py     -p outputs/train/dp_diffusion_ntt20/checkpoints/last/pretrained_model/     eval.n_episodes=10     eval.batch_size=1

