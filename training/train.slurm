#!/usr/bin/env bash
#SBATCH --job-name=pose # nom du job
#SBATCH --partition eaigc3_t # saturn-v
#SBATCH --nodes=1 # nombre de noeuds
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8 # nombre de GPU �|  réserver (un unique GPU ici)
#SBATCH --cpus-per-task=12
#SBATCH --time=500:00:00 # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=slurm_log/gan_1_steps_%j.out # nom du fichier de sortie
#SBATCH --error=slurm_log/gan_1_steps_%j.err # nom du fichier d'erreur (ici commun avec la sortie
#SBATCH --exclude=HOST-10-140-60-27
#SBATCH --quotatype=reserved

srun python train.py \
  --config_path configs/self_forcing_bi.yaml \
  --disable-wandb \
  --logdir "s3://wangyutong/ckpts/selfforcing-pose-1.3B-detail/"

# -x HOST-10-140-60-170,HOST-10-140-60-177
# srun --partition eaigc3_t --job-name=pose --nodes=2 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=12 -x HOST-10-140-60-93,HOST-10-140-60-102 python train.py --config_path configs/self_forcing_bi.yaml  --logdir "s3://wangyutong/ckpts/selfforcing-pose-1.3B-detail-lora32/" 

# python train.py --config_path configs/self_forcing_bi.yaml --disable-wandb --logdir "s3://wangyutong/ckpts/selfforcing-pose-1.3B-detail-lora32/" 

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=29065 train.py --config_path configs/self_forcing_bi.yaml --disable-wandb --logdir "s3://wangyutong/sf-vace/ckpts/short98k-full-cfg5/" 

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=29061 train.py --config_path configs/self_forcing_bi_gan.yaml --disable-wandb --logdir "s3://wangyutong/sf-vace/ckpts/short98k-otg+05dmd_alternate1_gan3-full-cfg3-gan01-dlr6e-7-eps02-norm/" 

# torchrun --nnodes=1 --nproc_per_node=3 --master_port=29061 train.py --config_path configs/self_forcing_bi_gan.yaml --disable-wandb --logdir "s3://wangyutong/sf-vace/ckpts/short98k-dmd_alternate1_gan3-ttur3-full-cfg3-gan01-glr2e-7-dlr6e-7-based_otg150/" 

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=29061 train.py --config_path configs/self_forcing_bi_gan.yaml --disable-wandb --logdir "s3://wangyutong/sf-vace/ckpts/5v2v_short98k-otg_alternate1_gan3-full-cfg3-gan01-glr1e-6-dlr6e-7-eps02-norm/" 

# torchrun --nnodes=1 --nproc_per_node=3 --master_port=29061 train.py --config_path configs/self_forcing_bi_gan.yaml --disable-wandb --logdir "s3://wangyutong/sf-vace/ckpts/5v2v_short98k-08otg+02dmd_alternate1_gan3-full-cfg3-gan01-glr1e-6-dlr6e-7-eps02-norm/" 


# torchrun --nnodes=1 --nproc_per_node=4 --master_port=29061 train.py --config_path configs/self_forcing_bi_gan.yaml --disable-wandb --logdir "s3://wangyutong/sf-vace/ckpts/alltask_short-08otg+02dmd_alternate1_gan3-full-cfg3-gan01-glr1e-6-dlr6e-7-eps02-norm/" 

# torchrun --nnodes=1 --nproc_per_node=3 --master_port=29061 train.py --config_path configs/self_forcing_bi_gan.yaml --disable-wandb --logdir "s3://wangyutong/sf-vace/ckpts/alltask_short-08otg+02dmd_alternate1_gan3-full-cfg3-gan01-glr1e-6-dlr6e-7-eps02-norm-freeze_accwan/" 
# torchrun --nnodes=1 --nproc_per_node=4 --master_port=29061 train.py --config_path configs/self_forcing_bi_gan.yaml --disable-wandb --logdir "s3://wangyutong/sf-vace/ckpts/alltask76k_short-08otg+02dmd_alternate1_gan3-cfg3-gan01-glr1e-6-dlr4e-7-ttur5-freeze_accwan/" 

# torchrun --nnodes=1 --nproc_per_node=4 --master_port=29061 train.py --config_path configs/self_forcing_bi_gan.yaml --disable-wandb --logdir "s3://wangyutong/sf-vace/ckpts/alltask76k_short-08otg+02dmd_alternate1_gan3-cfg3-glr1e-6-dlr4e-7-ttur5-accwan_init_1step/" 
