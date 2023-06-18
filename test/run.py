"""

"""
import time
import main
from test_image_sender import *
from send_image import *

# run fastApi frame stream server
main.run_server_fastApi()
time.sleep(10)

print('serwer uruchomiony')

print('\t http://localhost:6064/docs')
print('\n')
print('\t http://localhost:63342/fastapi-frame-stream/test/test_iframe_all.html')
print('\n')


# send test image
run_test_sender()
print('after run_test_sender')

# img_sender = ImageSender()

# test_send_image_file()
# print('\t http://localhost:63342/fastapi-frame-stream/test/test_iframe_all.html')
# time.sleep(30)


img_sender = ImageSender()
time.sleep(10)



print('koniec')
