# assmd
Adaptive Sampling Slurm Molecular Dynamics
TO INSTALL:
- download the git repo and put in directory on eagle where you have access
- using miniconda create env for ASSMD - using provided yml file
- activate conda env
- in main directory install ASSMD by "python -m pip install -e ."

TO USE:
- while not in interactive slurm job activate conda env with ASSMD installed
- python -m assmd -c [config.yml] -l [abs path for main log file]
- config options are explained in example_config.yml
