"""

"""
import os
import cv2
import cvb

vin_device = cvb.DeviceFactory.open(cvb.install_path() + "/drivers/GenICam.vin")
stream = vin_device.stream

device = cvb.DeviceFactory.open(os.path.join(cvb.install_path(), "drivers", "GenICam.vin"))
dev_node_map = device.node_maps["Device"]
id = dev_node_map["DeviceID"].value
dev_node_map["TriggerMode"].value = "On"
dev_node_map["TriggerSource"].value = "Software"

stream = device.stream()
stream.start()
stream_grabbed = 0
while stream_grabbed == 0:
    if TakeFrame:
        dev_node_map["ExposureTimeAbs"].value = self.expTime
        dev_node_map["TriggerSoftware"].execute()
        stream_frame, self.stream_grabbed = stream.wait_for(1000)
        self.frame = numpy.copy(cvb.as_array(stream_frame))
        self.TakeFrame = False
    else:
        time.sleep(0.00001)