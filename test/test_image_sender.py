"""Testing the ImageSender class from image_sender.py

"""
from image_sender import *
import pytest
from asgiref.sync import async_to_sync
import time


def test_send_image_file():
    img_sender = ImageSender()
    img_file = 'images\\chessboard_01.jpg'
    stream_id = 'test'
    print('test_send_image_file')

    @async_to_sync
    async def tt_send_frame_file():
        await img_sender.send_image_file(stream_id, img_file)
        time.sleep(0)
    result = False;
    try:
        tt_send_frame_file()
        result = True
    except:
        result = False;
    # time.sleep(30)
    assert result == True, "Powinno być True"


if __name__ == '__main__':
    test_send_image_file()

