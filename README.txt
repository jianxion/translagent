To run the code, first download bergsma500 dataset from the original translagent repo and change the src_path function in util.py to your own file path, then create a virtual environment using this command:
conda create -n myenv python=3.8 pytorch torchvision -c pytorch
conda activate myenv
Then run
python3 my_train.py --l1 en --l2 de --num_games 1 --batch_size 128 --stop_after 1 --cpu

Main problem needs to be resolved:
1. bergsma_words function from utils.py. Need to get access to the text file used in this function
