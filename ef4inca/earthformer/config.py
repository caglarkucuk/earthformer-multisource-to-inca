import os
from omegaconf import OmegaConf

_CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


cfg = OmegaConf.create()
cfg.root_dir = os.path.abspath(os.path.realpath(os.path.join(_CURR_DIR, "..", "..")))

## Directory to save results is changed for convenience! Note that there're other directory configuration done in 'train_cuboid_sevir_invLinear.py'!
cfg.exps_dir = os.path.join(cfg.root_dir, "experiments")  # default directory for saving experiment results
# cfg.exps_dir = "/home/kucuk/Results/EarthFormer"
# os.makedirs(cfg.exps_dir, exist_ok=True)
