import json
import os
import sys
from pathlib import Path
import fastapi
import uvicorn
from fastapi import Body

app = fastapi.FastAPI()


@app.get("/")
def read_root():
    return "这是一个训练服务器"


@app.get("/train/{fileUrl}")
def train(fileUrl: str = ""):
    pass


@app.get("/stop")
def stop():
    """
    stop the training

    """
    pass


@app.get("/trainConfigRoot")
def trainConfigRoot():
    return {
        "trainRootList": ["分类", "检测", "分割"],
        "frameList": {
            0: ["timm"],
            1: ["yolov5"],
            2: ["unet"]
        },
    }


@app.get("/trainConfig/{frame}")
def trainConfig(frame: str = ""):
    print(frame)
    if frame == "timm":
        return [
            {
                "title_":"配置文件路径",
                "key":"config",
                "type":"file",
                "value":"--",
            }
        ]


@app.post("/train")
def train(data: dict = Body(...)):
    # 处理 JSON 数据
    if data["frame"] == "timm":
        configFile = data["config"]
        trainRoot = Path(os.path.dirname(sys.executable) if "python.exe" not in sys.executable else os.path.dirname(__file__))
        trainPy = trainRoot/r"pytorch-image-models\train.py"
        cmd = f"python {trainPy} --config {configFile}"
        os.system(cmd)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
