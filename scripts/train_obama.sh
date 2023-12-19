python main.py data/obama/ --workspace trial_obama_triplane/ -O --iters 100000
cp -r trial_obama_triplane/checkpoints trial_obama_triplane/checkpoints_
python main.py data/obama/ --workspace trial_obama_triplane/ -O --iters 125000 --finetune_lips --patch_size 32
python main.py data/obama/ --workspace trial_obama_triplane/ -O --test
# python main.py data/obama/ --workspace trial_obama_triplane_torso/ -O --torso --head_ckpt <head>.pth --iters 200000