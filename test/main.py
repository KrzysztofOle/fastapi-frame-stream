import sys

import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
# from threading import Thread
import threading
import time
import asyncio

sys.path.append("./")

from fastapi_frame_stream import FrameStreamer
from send_image import send_image

app = FastAPI()
fs = FrameStreamer()

print('......')

def sending_images():
    print('START......')
    time.sleep(10)
    asyncio.run(send_image())
    time.sleep(1)


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

def server_fastApi():
    # start the fastApi app
    print('START fastAPI...')
    uvicorn.run(app, host="0.0.0.0", port=6064)

def run_server_fastApi():
    # Created the Threads
    th_server = threading.Thread(target=lambda: server_fastApi(), daemon=True)
    th_server.start()

def run_test_sender():
    # Created the Threads
    th_send = threading.Thread(target=sending_images, daemon=True)
    th_send.start()
    th_send.join()
    print('czekamy na koniec .............')
    time.sleep(20)
    print('.........    koniec')

# th_server.join()
# th_send.join()


if __name__ == '__main__':
    print('main: __name__ ...')
    # start the fastApi app
    run_server_fastApi()
    run_test_sender()
    print('main: __name__  END')
    # server_fastApi()
    # start_fastApiFS()


