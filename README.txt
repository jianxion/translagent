To run the code, first download multi30k dataset from the original translagent repo and change the scr_path(), multi30k_reorg_path() and coco_path() from /src/sentence/util.py
Then run
python3 rltrain.py --dataset multi30k --task 1 --rl --lr 0.00003 --num_games 20 --batch_size 128 --beam_width 4 --cpu

This will unfortunately produce an error as the dataloader is not fixed.
