from hydra import initialize, compose
from omegaconf import OmegaConf
import subprocess
from datetime import datetime

with initialize(config_path="configs"):
    cfg = compose(config_name="yolo")
    aug = compose(config_name="hyp")


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
    hsv_h={cfg.hsv_h} \
    hsv_s={cfg.hsv_s} \
    hsv_v={cfg.hsv_v} \
    degrees={cfg.degrees} \
    translate={cfg.translate} \
    scale={cfg.scale} \
    shear={cfg.shear} \
    perspective={cfg.perspective} \
    flipud={cfg.flipud} \
    fliplr={cfg.fliplr} \
    mosaic={cfg.mosaic} \
    mixup={cfg.mixup} \
    erasing={cfg.erasing} \
    name={run_name}"""

print(cmd)
subprocess.run("wandb login", shell=True, check=True)
subprocess.run("yolo settings wandb=True", shell=True, check=True)
subprocess.run(cmd, shell=True, check=True)
