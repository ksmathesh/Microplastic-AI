"""Fine-tune YOLOv8 on the supplied annotated dataset."""
import argparse,os
from pathlib import Path
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",default=os.getenv("DATASET_YAML",""))
    p.add_argument("--weights",default="yolov8n.pt")
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--device",default=None)
    a=p.parse_args(); data=Path(a.data).expanduser()
    if not data.is_file(): p.error("Set DATASET_YAML or pass --data <dataset data.yaml>.")
    import yaml
    config=yaml.safe_load(data.read_text(encoding="utf-8"))
    # Resolve Roboflow relative split paths against the actual dataset folder.
    config["path"]=str(data.parent.resolve())
    for split in ("train","val","test"):
        if split not in config: continue
        rel=Path(config[split])
        candidate=(data.parent/rel).resolve()
        if not candidate.exists() and ".." in rel.parts:
            candidate=(data.parent.joinpath(*[part for part in rel.parts if part != ".."])).resolve()
        if not candidate.exists(): candidate=(data.parent.parent/rel).resolve()
        config[split]=str(candidate)
    normalized=Path("runs/aquasentinel/dataset.normalized.yaml")
    normalized.parent.mkdir(parents=True,exist_ok=True)
    normalized.write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    from ultralytics import YOLO
    YOLO(a.weights).train(data=str(normalized.resolve()),epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,device=a.device,
        project="runs/aquasentinel",name="microplastic-yolov8n",exist_ok=True,plots=True,seed=42)
    print("Best checkpoint: runs/aquasentinel/microplastic-yolov8n/weights/best.pt")
if __name__=="__main__": main()



