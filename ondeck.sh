LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=js/ros_010 policy=diffusion_ros_real env=ros_real hydra.run.dir=outputs/train/dp_ros_010 hydra.job.name=dp_ros_010 device=cuda wandb.enable=false training.save_freq=20000

LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=js/ros_011 policy=diffusion_ros_real env=ros_real hydra.run.dir=outputs/train/dp_ros_011 hydra.job.name=dp_ros_011 device=cuda wandb.enable=false training.save_freq=20000

LEROBOT_HOME='' python lerobot/scripts/train.py dataset_repo_id=js/ros_009 policy=diffusion_ros_real env=ros_real hydra.run.dir=outputs/train/dp_ros_009 hydra.job.name=dp_ros_009 device=cuda wandb.enable=false training.save_freq=20000

