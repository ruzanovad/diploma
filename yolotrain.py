from hydra import initialize, compose
from omegaconf import OmegaConf
import subprocess
from datetime import datetime

with initialize(config_path="configs"):
    cfg = compose(config_name="yolo")


run_name = f"{cfg.name_prefix}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
cmd = f"""yolo train \
    model=yolo11n.pt \
    data={cfg.data} \
    epochs={cfg.epochs} \
    imgsz={cfg.imgsz} \
    seed={cfg.seed} \
    device={cfg.device} \
    batch={cfg.batch} \
    augment={cfg.augment} \
    rect={cfg.rect} \
    plots={cfg.plots} \
    visualize={cfg.visualize} \
    project={cfg.project} \
    hyp=configs/hyp.yaml \
    name={run_name}"""

print(cmd)
subprocess.run("wandb login", shell=True, check=True)
subprocess.run("yolo settings wandb=True", shell=True, check=True)
subprocess.run(cmd, shell=True, check=True)
