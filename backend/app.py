"""AquaSentinel image screening API."""
import json, os, sqlite3, uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT/"storage"; UPLOADS=STORE/"uploads"; RESULTS=STORE/"processed"
for folder in (UPLOADS,RESULTS): folder.mkdir(parents=True,exist_ok=True)
DB=Path(os.getenv("DATABASE_URL","sqlite:///./storage/aquasentinel.db").removeprefix("sqlite:///"))
if not DB.is_absolute(): DB=(ROOT/DB).resolve()
LIMIT=int(os.getenv("MAX_UPLOAD_MB","15"))
MODEL_PATH=Path(os.getenv("MODEL_PATH",str(ROOT/"weights"/"best.pt")))
MODEL: Any|None=None
NAMES={0:"Fibre",1:"Film",2:"Fragment",3:"Microbeads"}

def connect():
    DB.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(DB); db.row_factory=sqlite3.Row; return db

def init_db():
    with connect() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS analyses(
        id TEXT PRIMARY KEY,sample_name TEXT NOT NULL,source_file TEXT NOT NULL,result_file TEXT,
        image_width INTEGER,image_height INTEGER,particle_count INTEGER,classes_json TEXT,
        detections_json TEXT,mean_confidence REAL,sample_volume_ml REAL,created_at TEXT)""")

def load_model():
    global MODEL,NAMES
    if not MODEL_PATH.is_file(): return
    from ultralytics import YOLO
    MODEL=YOLO(str(MODEL_PATH))
    NAMES={int(k):str(v) for k,v in MODEL.names.items()}

@asynccontextmanager
async def lifespan(app):
    init_db()
    try: load_model()
    except Exception as exc: app.state.model_error=str(exc)
    yield

app=FastAPI(title="AquaSentinel AI",description="Image-based screening of suspected microplastic particles.",version="0.1.0",lifespan=lifespan)

def serialize(row):
    d=dict(row); d["classes"]=json.loads(d.pop("classes_json")); d["detections"]=json.loads(d.pop("detections_json"))
    d["source_url"]="/media/uploads/"+d["source_file"]
    d["result_url"]="/media/processed/"+d["result_file"] if d["result_file"] else None
    return d

@app.get("/api/status")
def status():
    return {"service":"AquaSentinel AI","model_ready":MODEL is not None,"model_file":MODEL_PATH.name if MODEL else None,
            "model_error":getattr(app.state,"model_error",None),"classes":list(NAMES.values()),
            "notice":"Detections indicate suspected particles and do not chemically identify plastic."}

@app.get("/api/analyses")
def analyses(limit:int=25):
    with connect() as db:
        rows=db.execute("SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?",(max(1,min(limit,100)),)).fetchall()
        total=db.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        count=db.execute("SELECT COALESCE(SUM(particle_count),0) FROM analyses").fetchone()[0]
    return {"items":[serialize(r) for r in rows],"total_analyses":total,"total_particles":count}

@app.post("/api/analyze")
async def analyze(image:UploadFile=File(...),sample_name:str=Form("Untitled sample"),sample_volume_ml:float|None=Form(None)):
    if MODEL is None: raise HTTPException(503,"No trained checkpoint. Follow README to train the detector and configure MODEL_PATH.")
    if sample_volume_ml is not None and sample_volume_ml<=0: raise HTTPException(422,"Sample volume must be greater than zero.")
    content=await image.read(LIMIT*1024*1024+1)
    if not content: raise HTTPException(400,"Choose an image to analyze.")
    if len(content)>LIMIT*1024*1024: raise HTTPException(413,f"Image exceeds {LIMIT} MB.")
    from io import BytesIO
    try:
        with Image.open(BytesIO(content)) as im: im.verify()
        with Image.open(BytesIO(content)) as im: rgb=im.convert("RGB"); width,height=rgb.size
    except (UnidentifiedImageError,OSError) as exc: raise HTTPException(415,"Upload a valid PNG, JPG, or WEBP image.") from exc
    if min(width,height)<32: raise HTTPException(422,"Image must be at least 32 × 32 pixels.")
    key=str(uuid.uuid4()); filename=key+".jpg"; src=UPLOADS/filename; out=RESULTS/filename; rgb.save(src,"JPEG",quality=95)
    try:
        pred=MODEL.predict(source=rgb,verbose=False)[0]; detections=[]
        if pred.boxes is not None:
            for box in pred.boxes:
                cid=int(box.cls.item()); coords=[round(float(v),2) for v in box.xyxy[0].tolist()]
                detections.append({"class_id":cid,"class_name":NAMES.get(cid,"Class "+str(cid)),"confidence":round(float(box.conf.item()),4),"box_xyxy":coords})
        classes=dict(Counter(x["class_name"] for x in detections))
        mean=sum(x["confidence"] for x in detections)/len(detections) if detections else None
        Image.fromarray(pred.plot()[:,:,::-1]).save(out,"JPEG",quality=92)
        created=datetime.now(timezone.utc).isoformat(); name=" ".join(sample_name.split())[:120] or "Untitled sample"
        with connect() as db:
            db.execute("INSERT INTO analyses VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",(key,name,filename,filename,width,height,len(detections),json.dumps(classes),json.dumps(detections),mean,sample_volume_ml,created))
            row=db.execute("SELECT * FROM analyses WHERE id=?",(key,)).fetchone()
        return serialize(row)
    except Exception as exc:
        src.unlink(missing_ok=True); out.unlink(missing_ok=True)
        raise HTTPException(500,"Inference failed: "+str(exc)) from exc

app.mount("/media/uploads",StaticFiles(directory=UPLOADS),name="uploads")
app.mount("/media/processed",StaticFiles(directory=RESULTS),name="processed")
app.mount("/",StaticFiles(directory=ROOT/"frontend",html=True),name="frontend")
