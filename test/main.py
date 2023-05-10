import sys

import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import threading
import time
import asyncio

sys.path.append("./")

from fastapi_frame_stream import FrameStreamer


app = FastAPI()
fs = FrameStreamer()

print('......')


class InputImg(BaseModel):
    img_base64str: str


@app.get('/')
def home():
    return {'ok'}


@app.post("/send_frame_from_string/{img_id}")
async def send_frame_from_string(img_id: str, d: InputImg):
    await fs.send_frame(img_id, d.img_base64str)


@app.post("/send_frame_from_file/{img_id}")
async def send_frame_from_file(img_id: str, file: UploadFile = File(...)):
    print('\n')
    print('send_frame_from_file ...')
    await fs.send_frame(img_id, file)


@app.get("/video_feed/{img_id}")
async def video_feed(img_id: str):
    return fs.get_stream(img_id)



if __name__ == '__main__':
    # start the flask app
    uvicorn.run(app, host="0.0.0.0", port=6064)


def server_fastApi():
    # start the fastApi app
    print('START fastAPI...')
    uvicorn.run(app, host="0.0.0.0", port=6064)


def run_server_fastApi():
    # Created the Threads
    th_server = threading.Thread(target=lambda: server_fastApi(), daemon=True)
    th_server.start()






"""
if __name__ == '__main__':
    print('main: __name__ ...')
    # start the fastApi app
    run_server_fastApi()
    run_test_sender()
    print('main: __name__  END')
    # server_fastApi()
    # start_fastApiFS()
"""

