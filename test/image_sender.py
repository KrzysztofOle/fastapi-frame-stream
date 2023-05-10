#!/usr/bin/env python
"""Defines the ImageSender class and it's interface with the in memory SQLite datebase
    The class is based on FrameStreamer by Tiago Prat (https://github.com/TiagoPrata/fastapi-frame-stream)

"""

from typing import Any, Mapping, Union
import cv2
import imutils
import time
import base64
import numpy as np
import sqlite3
from fastapi import BackgroundTasks, File, UploadFile
from starlette.datastructures import UploadFile as starletteUploadFile
from fastapi.responses import StreamingResponse
from fastapi_frame_stream import FrameStreamer


__author__ = "Tiago Prata"
__credits__ = ["Tiago Prata"]
__license__ = "MIT"
__version__ = "0.1.1"
__maintainer__ = "Tiago Prata"
__email__ = "prataaa@hotmail.com"
__status__ = "Beta version"


class ImageSender:
    """The ImageSender class allows you to send frames to FrameStreamer
    """
    __frame_streamer = None


    def __init__(self):
        try:
            self.__frame_streamer = FrameStreamer()

        except:
            print("Error ImageSender:__init__.")
            pass



    async def send_opencv_image(self, stream_id: str, img) -> None:
        """Send a frame to be streamed.

        Args:
            stream_id (str): ID (primary key) of the frame
            img: OpenCV image.
        """
        print('\t send_opencv_frame  stream_id: ', stream_id)
        jpg_img = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(jpg_img[1]).decode("utf-8")
        await self.__frame_streamer._store_image_str(stream_id, img_str)



    async def send_image_file(self, stream_id: str, file: str) -> None:
        """Send image local file to be streamed.

        Args:
            stream_id (str): ID (primary key) of the frame
            file (str): image file name
        """
        print('\t send_frame_file  stream_id: ', stream_id, '      file: ', file)
        img = cv2.imread(file)
        await self.send_opencv_image(stream_id, img)
        return img


    async def send_2image_file(self, stream_id: str, file: str) -> None:
        """Send a two frame to be streamed.

        Args:
            stream_id (str): ID (primary key) of the frame
            frame (Union[str, UploadFile, bytes]): Frame (image) to be streamed.
        """
        print('\t send_2frame_file  stream_id: ', stream_id, '      file: ', file)
        img = cv2.imread(file)
        await self.send_opencv_image(stream_id, img)
        stream_id_inv = stream_id + '_inv'
        img_inv = cv2.bitwise_not(img)
        await self.send_opencv_image(stream_id, img_inv)
