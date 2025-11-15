# Installation
Use the [following file](https://github.com/locuslab/get/blob/main/environment.yml) to set up the environment.

# Run
All the files needed for running the code are in `scripts`. The shell file to run 
1. training is `cmdline.sh` (which calls `train.py`) 
2. inference is `runeval.sh` (which calls `eval.py`)

An example run command on terminal
```
source cmdline.sh
```

# Necessary and Unnecessary folders
The folders which are used for the workshop submission are:
1. `models`
2. `scripts`
3. `utils`

Please ignore the other folders mentioned below because they were used for experimental ideas which were not included in the workshop submission. 
1. `diffusion`
2. `prng`
3. `quant`
4. `torch_utils`

# Key files
### models/models.py
The main code is adopted from the [DiT repo](https://github.com/facebookresearch/DiT/tree/main). Significant changes are done to implement multione which is described at various parts of the code through comments.

### scripts/train.py
This file contains all the arguments required for running training. It also calls the 4 main functions `get_data`, `get_models` , `get_trainsetup` and `train_batch`. It also runs the epoch-wise training for these models.

### utils/trainutils.py
Contains the functions `get_data`, `get_models` , `get_trainsetup` and `train_batch`. The first three are straight-forward, each providing dataloader, models (teacher and student) and the optimization setup. The last one has multiple training setups such `GET`, `DMD`, `Multione` and `Layer`. Each of them are explained with comments in the corresponding functions which are called.

# For building on top of this code
If you want to modify the model architecture, go and make the changes in `models/model.py`. If you want to modify other the training setup, go to `utils/trainutils.py` and change the `train_batch` function by adding a new if statement for your setup. Write the corresponding function below it.