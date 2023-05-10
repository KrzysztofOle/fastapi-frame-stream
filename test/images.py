import tkinter.filedialog
import numpy
import tkinter
import PIL.Image
import PIL.ImageTk
import datetime
import time
import abc
import cv2
import cvb
import os
from threading import Thread, Lock
from PIL import ImageDraw
import MTM
from pathlib import Path
from skimage import (feature, util)
from skimage.filters import (laplace, sobel, scharr, prewitt, farid, roberts, apply_hysteresis_threshold)
from skimage.morphology import (skeletonize, convex_hull_image, disk, reconstruction)
from skimage.filters.rank import entropy
from skimage.restoration import rolling_ball

import src.objects
import src.canvasfunction
import src.my_globals
import src.logers


class EthStreamHandler:
    """Handler for EthStream
    """

    def __init__(self):
        self.started = False
        self.TakeFrame = False
        self.frame = None
        self.expTime = None
        self.id = None
        self.connected = False
        self.thread = None

        self.stream_grabbed = None
        # self.stream_frame = None

    def start(self):
        if self.started:
            src.logers.log_info_q.put([__name__, 30, "Eth thread already started!!"])
            return None
        self.started = True
        self.thread = Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        import src.my_globals
        while self.started:
            try:
                with cvb.DeviceFactory.open(
                        os.path.join(cvb.install_path(), "drivers", "GenICam.vin")) as device:
                    dev_node_map = device.node_maps["Device"]
                    self.id = dev_node_map["DeviceID"].value
                    dev_node_map["TriggerMode"].value = "On"
                    dev_node_map["TriggerSource"].value = "Software"

                    stream = device.stream()
                    stream.start()
                    self.stream_grabbed = 0
                    while self.stream_grabbed == 0:
                        if self.TakeFrame:
                            dev_node_map["ExposureTimeAbs"].value = self.expTime
                            dev_node_map["TriggerSoftware"].execute()
                            stream_frame, self.stream_grabbed = stream.wait_for(1000)
                            self.frame = numpy.copy(cvb.as_array(stream_frame))
                            self.TakeFrame = False
                        else:
                            time.sleep(0.00001)
            except:
                self.TakeFrame = False
                src.logers.log_info_q.put([__name__, 40, 'No camera, reconnect in 5 seconds...'])
                self.frame = None
                time.sleep(5)

    def read(self, _exp_time):
        self.expTime = _exp_time
        self.TakeFrame = True
        while self.TakeFrame:
            time.sleep(0.0001)
            pass
        time.sleep(0.0001)
        return self.frame

    def stop(self):
        self.started = False
        self.thread.join()


class Images(metaclass=abc.ABCMeta):
    """An abstract class that holds all images
    """

    def __init__(self, name):
        """

        :param name: the name of the class instance
        """
        #: (str) the name of the class instance.
        self.name = name
        #: (ndarray) image of the class instance.
        self.img = None
        #: (dictionary) parameters of instance in tkinters variables
        self.inParameters = dict(name=tkinter.StringVar(), view=tkinter.BooleanVar(), log=tkinter.BooleanVar())

        self.inParameters['name'].set(src.my_globals.uniq_name(name))
        self.inParameters['view'].set(True)
        self.inParameters['log'].set(False)

    @abc.abstractmethod
    def do(self):
        """Abstract method
        """
        pass

    def update(self):
        """Wrapper to run all function

        :return: image
        """
        try:
            self.do()
            if self.inParameters['log'].get():
                src.logers.log_image_q.put([self.inParameters['name'].get(), numpy.copy(self.img)])
        except BaseException as error:
            src.logers.log_info_q.put([__name__, 40, 'An exception occurred: {} in '.format(error) + self.name])
        if src.my_globals.bView.get() and self.inParameters['view'].get():
            try:
                src.canvasfunction.update_view(0, self)
            except BaseException as error:
                src.logers.log_info_q.put([__name__, 40, 'An exception occurred: {} in '.format(error) + self.name])
                src.logers.log_info_q.put([__name__, 10, 'Image not available'])

        if self.inParameters['view'].get():
            return self.img
        else:
            return None

    @abc.abstractmethod
    def stop(self):
        """Abstract method
        """
        pass

    @staticmethod
    def binarize_image(_img):
        _img[_img < 255] = 0
        _img[_img == 255] = 1
        return _img

    @staticmethod
    def float_bound(_img):
        _img[_img > 1] = 1
        _img[_img < -1] = -1
        return _img

    @staticmethod
    def bin_to_byte_image(_img):
        _img[_img > 0] = 255
        return _img


class ImageTransform(Images):
    """Images Subclass, is subclassed by image transformation subcalass: ITPerspective, ITMask, ITCrop, ITAddObject,
    ITShowLabels, ITRotate, ITFlip
    """

    def __init__(self, name, source_name=None):
        """

        :param name: the name of the class instance
        :param source_name: the name of the input image instance
        """
        #: (ndarray)input image
        self.img_source = None

        super().__init__(name)

        # noinspection PyTypeChecker
        self.inParameters['SourceImage'] = tkinter.StringVar()
        if source_name is None:
            self.inParameters['SourceImage'].set("None")
        else:
            self.inParameters['SourceImage'].set(source_name)

    @abc.abstractmethod
    def do(self):
        pass

    def stop(self):
        pass


class ImageFilter(Images):
    """Images Subclass, is subclassed by image filter subcalass: IFGaussianBlur, IFSharp, IFLog, IFErosion,
    IFDilation, IFOpening, IFClosing, IFLaplacian, IFThreshold, IFBitwiseNot
    """

    def __init__(self, name, source_name=None):
        """

        :param name: the name of the class instance
        :param source_name: the name of the input image instance
        """
        #: (ndarray)input image
        self.img_source = None

        super().__init__(name)
        # noinspection PyTypeChecker
        self.inParameters['SourceImage'] = tkinter.StringVar()
        if source_name is None:
            self.inParameters['SourceImage'].set("None")
        else:
            self.inParameters['SourceImage'].set(source_name)

        self.inParameters['Object1'] = tkinter.StringVar()
        self.inParameters['Object1'].set("None")

    def do(self):
        self.img = numpy.copy(util.img_as_ubyte(src.my_globals.get_from_list(src.my_globals.oneList,
                                                                             name=self.inParameters[
                                                                                 'SourceImage'].get(),
                                                                             img=True)))

        roi_name = self.inParameters['Object1'].get()
        if roi_name != 'None':
            obj1 = src.my_globals.get_from_list(src.my_globals.oneList, name=self.inParameters['Object1'].get())
            y0 = int(obj1.y - obj1.inParameters['height'].get() / 2)
            y1 = int(obj1.y + obj1.inParameters['height'].get() / 2)
            x0 = int(obj1.x - obj1.inParameters['width'].get() / 2)
            x1 = int(obj1.x + obj1.inParameters['width'].get() / 2)
        else:
            y0 = 0
            y1 = self.img.shape[0]
            x0 = 0
            x1 = self.img.shape[1]

        roi = self.img[y0:y1, x0:x1]
        self.img[y0:y1, x0:x1] = util.img_as_ubyte(self.filter_roi(roi))

    @abc.abstractmethod
    def filter_roi(self, img):
        pass

    def stop(self):
        pass


# class MargeTo4(Images):
#     """
#         *dev
#
#     """

#     def __init__(self, name, index=None):
#         super().__init__(name)
#
#         # noinspection PyTypeChecker
#         self.inParameters['SourceImage1.index'] = tkinter.IntVar()
#         # noinspection PyTypeChecker
#         self.inParameters['SourceImage2.index'] = tkinter.IntVar()
#         # noinspection PyTypeChecker
#         self.inParameters['SourceImage3.index'] = tkinter.IntVar()
#         # noinspection PyTypeChecker
#         self.inParameters['SourceImage4.index'] = tkinter.IntVar()
#         if index is None:
#             self.inParameters['SourceImage1.index'].set(0)
#             self.inParameters['SourceImage2.index'].set(0)
#             self.inParameters['SourceImage3.index'].set(0)
#             self.inParameters['SourceImage4.index'].set(0)
#         else:
#             self.inParameters['SourceImage1.index'].set(index)
#             self.inParameters['SourceImage2.index'].set(index)
#             self.inParameters['SourceImage3.index'].set(index)
#             self.inParameters['SourceImage4.index'].set(index)
#
#         self.img_source1 = None
#         self.img_source2 = None
#         self.img_source3 = None
#         self.img_source4 = None
#
#     def do(self):
#         src.my_globals.oneList[self.inParameters['SourceImage1.index'].get()].do()
#         self.img_source1 = src.my_globals.oneList[self.inParameters['SourceImage1.index'].get()].img
#         src.my_globals.oneList[self.inParameters['SourceImage2.index'].get()].do()
#         self.img_source2 = src.my_globals.oneList[self.inParameters['SourceImage2.index'].get()].img
#         src.my_globals.oneList[self.inParameters['SourceImage3.index'].get()].do()
#         self.img_source3 = src.my_globals.oneList[self.inParameters['SourceImage3.index'].get()].img
#         src.my_globals.oneList[self.inParameters['SourceImage4.index'].get()].do()
#         self.img_source4 = src.my_globals.oneList[self.inParameters['SourceImage4.index'].get()].img
#         img12 = cv2.hconcat([self.img_source1, self.img_source2])
#         img34 = cv2.hconcat([self.img_source3, self.img_source4])
#         # self.img = img13
#         self.img = cv2.vconcat([img12, img34])


class ITPerspective(ImageTransform):
    """Make image from another using 4 objects
    `OpenCV warpPerspective() <https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html \
    #gaf73673a7e8e18ec6963e3774e6a94b87>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)

        # noinspection PyTypeChecker
        self.inParameters['Object1'] = tkinter.StringVar()
        # noinspection PyTypeChecker
        self.inParameters['Object2'] = tkinter.StringVar()
        # noinspection PyTypeChecker
        self.inParameters['Object3'] = tkinter.StringVar()
        # noinspection PyTypeChecker
        self.inParameters['Object4'] = tkinter.StringVar()

        # noinspection PyTypeChecker
        self.inParameters['padding.left'] = tkinter.DoubleVar()
        self.inParameters['padding.left'].set(0)
        # noinspection PyTypeChecker
        self.inParameters['padding.right'] = tkinter.DoubleVar()
        self.inParameters['padding.right'].set(0)
        # noinspection PyTypeChecker
        self.inParameters['padding.top'] = tkinter.DoubleVar()
        self.inParameters['padding.top'].set(0)
        # noinspection PyTypeChecker
        self.inParameters['padding.bottom'] = tkinter.DoubleVar()
        self.inParameters['padding.bottom'].set(0)
        # noinspection PyTypeChecker
        self.inParameters['output.width'] = tkinter.IntVar()
        self.inParameters['output.width'].set(0)
        # noinspection PyTypeChecker
        self.inParameters['output.height'] = tkinter.IntVar()
        self.inParameters['output.height'].set(0)

    def do(self):
        """Main function
        """
        self.img_source = src.my_globals.get_from_list(src.my_globals.oneList,
                                                       name=self.inParameters['SourceImage'].get(),
                                                       img=True)
        obj1 = src.my_globals.get_from_list(src.my_globals.oneList, name=self.inParameters['Object1'].get())
        obj2 = src.my_globals.get_from_list(src.my_globals.oneList, name=self.inParameters['Object2'].get())
        obj3 = src.my_globals.get_from_list(src.my_globals.oneList, name=self.inParameters['Object3'].get())
        obj4 = src.my_globals.get_from_list(src.my_globals.oneList, name=self.inParameters['Object4'].get())

        input_transform = numpy.float32([[obj1.x, obj1.y],
                                         [obj2.x, obj2.y],
                                         [obj3.x, obj3.y],
                                         [obj4.x, obj4.y]])

        output_transform = numpy.float32(
            [[self.inParameters['padding.left'].get(), self.inParameters['padding.top'].get()],
             [self.inParameters['output.width'].get() + self.inParameters['padding.left'].get(),
              self.inParameters['padding.top'].get()],
             [self.inParameters['output.width'].get() + self.inParameters['padding.left'].get(),
              self.inParameters['output.height'].get() + self.inParameters['padding.top'].get()],
             [self.inParameters['padding.left'].get(), self.inParameters['output.height'].get()
              + self.inParameters['padding.top'].get()]])

        matrix = cv2.getPerspectiveTransform(input_transform, output_transform)

        self.img = cv2.warpPerspective(self.img_source,
                                       matrix,
                                       (int(self.inParameters['output.width'].get()
                                            + self.inParameters['padding.left'].get()
                                            + self.inParameters['padding.right'].get()),
                                        int(self.inParameters['output.height'].get()
                                            + self.inParameters['padding.bottom'].get()
                                            + self.inParameters['padding.top'].get())),
                                       cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=(0, 0, 0))


class ITMask(ImageTransform):
    """Mask image using object
    """

    def __init__(self, name, source_name=None):

        #: (ndarray)mask from object
        self.mask = None
        super().__init__(name, source_name)
        self.inParameters['Object1'] = tkinter.StringVar()
        self.inParameters['Object1'].set("None")
        # noinspection PyTypeChecker
        self.inParameters['inside'] = tkinter.BooleanVar()
        self.inParameters['inside'].set(True)

    def do(self):
        """Main function
        """
        self.img_source = src.my_globals.get_from_list(src.my_globals.oneList,
                                                       name=self.inParameters['SourceImage'].get(),
                                                       img=True)
        obj1 = src.my_globals.get_from_list(src.my_globals.oneList, name=self.inParameters['Object1'].get())
        self.mask = numpy.zeros(self.img_source.shape[:2], dtype='uint8')
        color = (255, 255, 255)
        if isinstance(obj1, src.objects.Rectangle):
            pts = numpy.array(obj1.newxy, numpy.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(self.mask, pts=[pts], color=color)
        elif isinstance(obj1, src.objects.Circle):
            self.mask = cv2.circle(self.mask, (int(obj1.x), int(obj1.y)), int(obj1.radius), color, -1)

        if not self.inParameters['inside'].get():
            self.mask = cv2.bitwise_not(self.mask)

        self.img = cv2.bitwise_and(self.img_source, self.img_source, mask=self.mask)


class ITCrop(ImageTransform):
    """Mask image using object
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        self.inParameters['Object1'] = tkinter.StringVar()
        self.inParameters['Object1'].set("None")

    def do(self):
        self.img_source = src.my_globals.get_from_list(src.my_globals.oneList,
                                                       name=self.inParameters['SourceImage'].get(),
                                                       img=True)
        obj1 = src.my_globals.get_from_list(src.my_globals.oneList, name=self.inParameters['Object1'].get())
        self.img = self.img_source[
                   int(obj1.y) - int(obj1.inParameters['height'].get() / 2):
                   int(obj1.y) + int(obj1.inParameters['height'].get() / 2),
                   int(obj1.x) - int(obj1.inParameters['width'].get() / 2):
                   int(obj1.x) + int(obj1.inParameters['width'].get() / 2)]


class ITAddObject(ImageTransform):
    """Add object to image
    """

    def __init__(self, name, source_name=None):
        #: image to PIL
        self.imgPil = None
        #: PIL drawing interface
        self.imgDraw = None

        super().__init__(name, source_name)
        self.inParameters['Object1'] = tkinter.StringVar()
        self.inParameters['Object1'].set("None")

    def do(self):
        """Main function
        """
        self.img_source = src.my_globals.get_from_list(src.my_globals.oneList,
                                                       name=self.inParameters['SourceImage'].get(),
                                                       img=True)
        self.imgPil = PIL.Image.fromarray(self.img_source)
        obj1 = src.my_globals.get_from_list(src.my_globals.oneList, name=self.inParameters['Object1'].get())
        src.my_globals.canvas.itemconfigure(obj1.inParameters['name'].get(), state=tkinter.NORMAL)

        self.imgDraw = ImageDraw.Draw(self.imgPil)
        self.imgDraw.polygon(((obj1.newxy[0], obj1.newxy[1]), (obj1.newxy[2], obj1.newxy[3]),
                              (obj1.newxy[4], obj1.newxy[5]), (obj1.newxy[6], obj1.newxy[7])), fill="white")
        self.img = numpy.array(self.imgPil)


class ITShowLabels(ImageTransform):
    """Add labels for src.detector.MultiTemplateMatching
    """

    def __init__(self, name, source_name=None):
        #: labels with scoring
        self.labels = None
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['SourceDetector'] = tkinter.StringVar()
        if source_name is None:
            self.inParameters['SourceDetector'].set("None")
        else:
            self.inParameters['SourceDetector'].set(source_name)

    def do(self):
        """Main function
        """
        self.img_source = src.my_globals.get_from_list(src.my_globals.oneList,
                                                       name=self.inParameters['SourceImage'].get(),
                                                       img=True)
        self.labels = src.my_globals.get_from_list(src.my_globals.oneList,
                                                   name=self.inParameters['SourceDetector'].get()).Hits

        self.img = MTM.drawBoxesOnRGB(self.img_source, self.labels, showLabel=True)


class ITRotate(ImageTransform):
    """`OpenCV rotate() <https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga4ad01c0978b0ce64baa246811deeac24>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['RotateFlags'] = tkinter.IntVar()
        self.inParameters['RotateFlags'].set(0)

    def do(self):
        """Main function
        """
        self.img_source = src.my_globals.get_from_list(src.my_globals.oneList,
                                                       name=self.inParameters['SourceImage'].get(),
                                                       img=True)
        self.img = cv2.rotate(self.img_source, self.inParameters['RotateFlags'].get())


class ITFlip(ImageTransform):
    """`OpenCV flip() <https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['FlipFlags'] = tkinter.IntVar()
        self.inParameters['FlipFlags'].set(0)

    def do(self):
        """Main function
        """
        self.img_source = src.my_globals.get_from_list(src.my_globals.oneList,
                                                       name=self.inParameters['SourceImage'].get(),
                                                       img=True)
        self.img = cv2.flip(self.img_source, self.inParameters['FlipFlags'].get())


class ITArithmeticOp(ImageTransform):
    """`OpenCV add, substract etc.
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['AoFlags'] = tkinter.IntVar()
        self.inParameters['AoFlags'].set(0)
        self.inParameters['SourceImage2'] = tkinter.StringVar()
        self.inParameters['SourceImage2'].set('None')

    def do(self):
        """Main function
        """
        self.img_source = src.my_globals.get_from_list(src.my_globals.oneList,
                                                       name=self.inParameters['SourceImage'].get(),
                                                       img=True)
        self.img_source2 = src.my_globals.get_from_list(src.my_globals.oneList,
                                                        name=self.inParameters['SourceImage2'].get(),
                                                        img=True)
        filtr = self.inParameters['AoFlags'].get()
        if filtr == 0:
            self.img = cv2.add(self.img_source, self.img_source2)
        elif filtr == 1:
            self.img = cv2.subtract(self.img_source, self.img_source2)
        elif filtr == 2:
            self.img = cv2.bitwise_and(self.img_source, self.img_source2)
        elif filtr == 3:
            self.img = cv2.bitwise_or(self.img_source, self.img_source2)
        elif filtr == 4:
            self.img = cv2.bitwise_xor(self.img_source, self.img_source2)


class IFGaussianBlur(ImageFilter):
    """`OpenCV GaussianBlur() <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['kSize.width'] = tkinter.IntVar()
        self.inParameters['kSize.width'].set(5)
        # noinspection PyTypeChecker
        self.inParameters['kSize.height'] = tkinter.IntVar()
        self.inParameters['kSize.height'].set(5)

    def filter_roi(self, _img):
        """Main function
        """

        return cv2.GaussianBlur(_img, (self.inParameters['kSize.width'].get(), self.inParameters['kSize.height'].get()),
                                0)


class IFSharp(ImageFilter):
    """Sharpened image by subtraction image and blured image
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['kSize.width'] = tkinter.IntVar()
        self.inParameters['kSize.width'].set(5)
        # noinspection PyTypeChecker
        self.inParameters['kSize.height'] = tkinter.IntVar()
        self.inParameters['kSize.height'].set(5)
        # noinspection PyTypeChecker
        self.inParameters['ratio'] = tkinter.DoubleVar()
        self.inParameters['ratio'].set(2)

    def filter_roi(self, _img):
        """Main function
        """
        blured = cv2.GaussianBlur(_img,
                                  (self.inParameters['kSize.width'].get(), self.inParameters['kSize.height'].get()), 0)
        return cv2.addWeighted(_img, self.inParameters['ratio'].get(), blured, -1, 0)


class IFLog(ImageFilter):
    """Image log transformation
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['maxLog'] = tkinter.IntVar()
        self.inParameters['maxLog'].set(255)

    def filter_roi(self, _img):
        """Main function
        """
        c = 255 / numpy.log10(1 + self.inParameters['maxLog'].get())
        _img[_img < 255] += 1
        img_log = c * numpy.log10(_img)
        return numpy.array(img_log, dtype=numpy.uint8)


class IFMorph(ImageFilter):
    """`OpenCV morphologyEx() <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['kSize.width'] = tkinter.IntVar()
        self.inParameters['kSize.width'].set(5)
        # noinspection PyTypeChecker
        self.inParameters['kSize.height'] = tkinter.IntVar()
        self.inParameters['kSize.height'].set(5)
        # noinspection PyTypeChecker
        self.inParameters['MorphType'] = tkinter.IntVar()
        self.inParameters['MorphType'].set(0)
        # noinspection PyTypeChecker
        self.inParameters['iterations'] = tkinter.IntVar()
        self.inParameters['iterations'].set(1)

    def filter_roi(self, _img):
        """Main function
        """
        kernel = numpy.ones((self.inParameters['kSize.width'].get(), self.inParameters['kSize.height'].get()),
                            numpy.uint8)
        return cv2.morphologyEx(_img, self.inParameters['MorphType'].get(), kernel,
                                iterations=self.inParameters['iterations'].get())


class IFLaplacian(ImageFilter):
    """`OpenCV Laplacian() <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html \
    #gad78703e4c8fe703d479c1860d76429e6>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['kSize'] = tkinter.IntVar()
        self.inParameters['kSize'].set(5)
        self.inParameters['var'] = tkinter.IntVar()
        self.inParameters['var'].set(0)

    def filter_roi(self, _img):
        """Main function
        """
        img = Images.float_bound(laplace(_img, ksize=self.inParameters['kSize'].get()))
        self.inParameters['var'].set(img.var())
        return img


class IFThreshold(ImageFilter):
    """`OpenCV threshold() <https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['Threshold'] = tkinter.IntVar()
        self.inParameters['Threshold'].set(20)
        # noinspection PyTypeChecker
        self.inParameters['ThreshMax'] = tkinter.IntVar()
        self.inParameters['ThreshMax'].set(255)
        # noinspection PyTypeChecker
        self.inParameters['ThreshType'] = tkinter.IntVar()
        self.inParameters['ThreshType'].set(0)

    def filter_roi(self, _img):
        """Main function
        """
        ret, roi_img = cv2.threshold(_img, self.inParameters['Threshold'].get(),
                                     self.inParameters['ThreshMax'].get(),
                                     self.inParameters['ThreshType'].get())
        return roi_img


class IFSkeletonize(ImageFilter):
    """`Scikit Skeletonize <https://scikit-image.org/docs/dev/auto_examples/edges/plot_skeleton.html#sphx-glr-auto-examples-edges-plot-skeleton-py>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker

    def filter_roi(self, _img):
        """Main function
        """
        return skeletonize(Images.binarize_image(_img))


class IFDoG(ImageFilter):
    """`OpenCV GaussianBlur() <https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['low_kSize.width'] = tkinter.IntVar()
        self.inParameters['low_kSize.width'].set(3)
        # noinspection PyTypeChecker
        self.inParameters['low_kSize.height'] = tkinter.IntVar()
        self.inParameters['low_kSize.height'].set(3)
        # noinspection PyTypeChecker
        self.inParameters['high_kSize.width'] = tkinter.IntVar()
        self.inParameters['high_kSize.width'].set(5)
        # noinspection PyTypeChecker
        self.inParameters['high_kSize.height'] = tkinter.IntVar()
        self.inParameters['high_kSize.height'].set(5)

    def filter_roi(self, _img):
        """Main function
        """
        low = cv2.GaussianBlur(_img, (
            self.inParameters['low_kSize.width'].get(), self.inParameters['low_kSize.height'].get()), 0)
        high = cv2.GaussianBlur(_img, (
            self.inParameters['high_kSize.width'].get(), self.inParameters['high_kSize.height'].get()), 0)
        return low - high


class IFCannyEdge(ImageFilter):
    """`Scikit CannyEdge <https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html#sphx-glr-auto-examples-edges
    -plot-canny-py>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['Sigma'] = tkinter.IntVar()
        self.inParameters['Sigma'].set(1)

    def filter_roi(self, _img):
        """Main function
        """
        return feature.canny(_img, self.inParameters['Sigma'].get())


class IFSobel(ImageFilter):
    """`Scikit Sobel <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.sobel>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)

    def filter_roi(self, _img):
        """Main function
        """

        return sobel(_img)


class IFScharr(ImageFilter):
    """`Scikit Scharr <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.scharr>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)

    def filter_roi(self, _img):
        """Main function
        """

        return scharr(_img)


class IFPrewitt(ImageFilter):
    """`Scikit Prewitt <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.prewitt>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)

    def filter_roi(self, _img):
        """Main function
        """

        return prewitt(_img)


class IFFarid(ImageFilter):
    """`Scikit Farid <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.farid>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)

    def filter_roi(self, _img):
        """Main function
        """

        return farid(_img)


class IFRoberts(ImageFilter):
    """`Scikit Roberts <https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.roberts>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)

    def filter_roi(self, _img):
        """Main function
        """

        return roberts(_img)


class IFConvexHull(ImageFilter):
    """`Scikit Convex Hull <https://scikit-image.org/docs/dev/auto_examples/edges/plot_convex_hull.html#sphx-glr-auto-examples-edges-plot-convex-hull-py>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)

    def filter_roi(self, _img):
        """Main function
        """
        return convex_hull_image(Images.binarize_image(_img))


class IFEntropy(ImageFilter):
    """`Scikit Entropy <https://scikit-image.org/docs/dev/api/skimage.filters.rank.html#skimage.filters.rank.entropy>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['radius'] = tkinter.IntVar()
        self.inParameters['radius'].set(100)

    def filter_roi(self, _img):
        """Main function
        """
        return Images.float_bound(entropy(_img, disk(self.inParameters['disk_radius'].get())))


class IFRollingBall(ImageFilter):
    """`Scikit RollingBall<https://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration
    .rolling_ball>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['disk_radius'] = tkinter.IntVar()
        self.inParameters['disk_radius'].set(5)

    def filter_roi(self, _img):
        """Main function
        """
        background = rolling_ball(_img, radius=self.inParameters['disk_radius'].get())
        return _img - background


class IFRegionalMaxima(ImageFilter):
    """`Scikit Regional Maxima <https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_regional_maxima.html
    #sphx-glr-auto-examples-color-exposure-plot-regional-maxima-py>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)
        # noinspection PyTypeChecker
        self.inParameters['h'] = tkinter.DoubleVar()
        self.inParameters['h'].set(0.3)

    def filter_roi(self, _img):
        """Main function
        """

        seed = _img - self.inParameters['h'].get()
        dilated = reconstruction(seed, _img, method='dilation')
        return _img - dilated


class IFHysteresisThresholding(ImageFilter):
        """`Scikit Hysteresis thresholding <https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html#sphx-glr-auto-examples-filters-plot-hysteresis-py>`_
        """

        def __init__(self, name, source_name=None):
            super().__init__(name, source_name)
            # noinspection PyTypeChecker
            self.inParameters['low'] = tkinter.DoubleVar()
            self.inParameters['low'].set(40)
            self.inParameters['high'] = tkinter.DoubleVar()
            self.inParameters['high'].set(100)

        def filter_roi(self, _img):
            """Main function
            """

            hight = (_img > self.inParameters['high'].get()).astype(int)
            hyst = apply_hysteresis_threshold(_img, self.inParameters['low'].get(), self.inParameters['high'].get())
            eq = hight + hyst
            return ImageFromEth.bin_to_byte_image(eq)


class IFBitwiseNot(ImageFilter):
    """`OpenCV bitwise_not() <https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga0002cf8b418479f4cb49a75442baee2f>`_
    """

    def __init__(self, name, source_name=None):
        super().__init__(name, source_name)

    def filter_roi(self, _img):
        """Main function
        """
        return cv2.bitwise_not(_img)


class ImageFromFile(Images):
    """Open image from file
    """

    def __init__(self, name):
        super().__init__(name)
        self.inParameters['path'] = tkinter.StringVar()

    def do(self):
        path = self.inParameters['path'].get()
        if len(path) <= 0:
            self.inParameters['path'].set(tkinter.filedialog.askopenfilename())
            path = self.inParameters['path'].get()
        if len(path) > 0:
            self.img = cv2.imread(path)
        try:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        except:
            pass
        self.inParameters['height'] = self.img.shape[0]
        self.inParameters['width'] = self.img.shape[1]

    def stop(self):
        pass


class ImageFromUsbCamera(Images):
    """Capture image from USB Camera
    """

    def __init__(self, name):
        super().__init__(name)
        #: USB camera
        self.cameraUSB = None
        # noinspection PyTypeChecker
        self.inParameters['port'] = tkinter.IntVar()
        self.inParameters['port'].set(0)

    def do(self):
        self.cameraUSB = cv2.VideoCapture(self.inParameters['port'].get(), cv2.CAP_DSHOW)
        ret, self.img = self.cameraUSB.read()
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.inParameters['height'] = self.img.shape[0]
        self.inParameters['width'] = self.img.shape[1]
        self.cameraUSB.release()
        return ret

    def stop(self):
        pass


class ImageFromEth(Images):
    """Class to use GigE Vision camera with `CVB <https://www.commonvisionblox.com/en>`_
    """
    #: (bool) thread mode started
    vsHandler = EthStreamHandler().start()
    #: instance od EthStreamHandler - thread to capture image
    imgThreadStarted = True
    #: 3x3 transformation matrix for cv.warpPerspective
    h = None

    def __init__(self, name):
        #: (bool) thread mode started
        # self.imgThreadStarted = False
        #: instance od EthStreamHandler - thread to capture image
        # self.vsHandler = None
        #: (ndarray) raw data from camera
        self.img_source = None
        #: 3x3 transformation matrix for cv.warpPerspective
        # self.h = None
        super().__init__(name)
        # noinspection PyTypeChecker
        self.inParameters['Exposure'] = tkinter.IntVar()
        self.inParameters['Exposure'].set(10)
        self.inParameters['ID'] = tkinter.IntVar()
        self.inParameters['ID'].set(0)
        self.inParameters['Perspective'] = tkinter.BooleanVar()
        self.inParameters['Perspective'].set(False)
        self.inParameters['PerspectiveWidth'] = tkinter.IntVar()
        self.inParameters['PerspectiveWidth'].set(1600)
        self.inParameters['PerspectiveHeight'] = tkinter.IntVar()
        self.inParameters['PerspectiveHeight'].set(1600)
        self.inParameters["DistortionCalibrated"] = False
        self.inParameters['Calibration'] = tkinter.BooleanVar()
        self.inParameters['Calibration'].set(False)
        self.inParameters['SourceDetector'] = tkinter.StringVar()
        self.inParameters['SourceDetector'].set('None')
        self.inParameters['calibX'] = tkinter.IntVar()
        self.inParameters['calibX'].set(9)
        self.inParameters['calibY'] = tkinter.IntVar()
        self.inParameters['calibY'].set(9)
        self.inParameters['distancePx'] = tkinter.IntVar()
        self.inParameters['distancePx'].set(200)
        self.check_calibration_file()

    def do(self):
        """Main function, its capture image, crop to square, call calibration make perspective using H transformation
        matrix
        """

        # img_souce
        self.check_calibration_file()
        self.img_source = numpy.copy(ImageFromEth.vsHandler.read(self.inParameters['Exposure'].get() * 1000))

        # img_calibrated
        if self.inParameters['Calibration'].get():
            self.calibrate()

        if self.inParameters["DistortionCalibrated"]:
            # img_perspective
            if self.inParameters['Perspective'].get():
                self.img = cv2.warpPerspective(self.img_source, ImageFromEth.h,
                                               (self.inParameters['PerspectiveWidth'].get(),
                                                self.inParameters['PerspectiveHeight'].get()))
            else:
                self.img = self.img_source
        else:
            self.img = self.img_source

        self.inParameters['height'] = self.img.shape[0]
        self.inParameters['width'] = self.img.shape[1]

    def stop(self):
        """Stop vsHandler
        """
        if ImageFromEth.imgThreadStarted:
            ImageFromEth.vsHandler.stop()
            ImageFromEth.imgThreadStarted = False

    def check_calibration_file(self):
        """Use H transformation matrix if file for used camera exist.
        """
        if ImageFromEth.vsHandler.id is not None and not self.inParameters["DistortionCalibrated"]:
            cam_id = ImageFromEth.vsHandler.id
            dirname = os.path.dirname(__file__)
            filename = os.path.join(dirname, '../data/' + cam_id + '.npy')
            if os.path.exists(filename):
                src.my_globals.cameras[cam_id] = src.my_globals.load_cameras(cam_id, filename)
            if cam_id in src.my_globals.cameras:
                ImageFromEth.h = src.my_globals.cameras[cam_id]['H']
                self.inParameters["DistortionCalibrated"] = True

    def calibrate(self):
        """Generate calibration file camera_id.npy with H transformation matrix.
        H transformation is crate using `cv2.findHomography() <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780>`_
        It finds perspective transformation between x/y grid and objects find by SourceDetector
        """
        detector1 = src.my_globals.get_from_list(src.my_globals.oneList, name=self.inParameters['SourceDetector'].get())

        x = self.inParameters['calibX'].get()
        y = self.inParameters['calibY'].get()

        corners = []

        checkerboard = (x, y)
        objectp3d = numpy.zeros((checkerboard[0] * checkerboard[1], 1, 2), numpy.float32)
        objectp3d[:, 0, :2] = numpy.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2) * self.inParameters[
            'distancePx'].get()
        for obj in src.my_globals.instance_list(src.my_globals.oneList, src.my_globals.CLASS_DICT.get('Object')):
            try:
                if obj.parent == detector1.inParameters["name"].get():
                    corners.append([obj.x, obj.y])
            except:
                src.logers.log_info_q.put([__name__, 40, 'error in calibration'])
        corners = numpy.array(corners, numpy.float32)

        sorted_corners = corners[corners[:, 1].argsort()]
        for n in range(0, x):
            sli = sorted_corners[n * x:n * x + y]
            sorted_corners[n * x:n * x + y] = sli[sli[:, 0].argsort()]
        corners = numpy.expand_dims(sorted_corners, axis=1)

        h, mask = cv2.findHomography(corners, objectp3d)
        # cam_id = self.inParameters['hide_DeviceID']
        cam_id = ImageFromEth.vsHandler.id
        src.my_globals.cameras[cam_id] = {'H': h}
        ImageFromEth.h = src.my_globals.cameras[cam_id]['H']
        src.my_globals.save_cameras(cam_id)
        self.check_calibration_file()


#
# class RmaVS(Images):
#     def __init__(self, name):
#         super().__init__(name)
#         # noinspection PyTypeChecker
#         self.inParameters['Exposure'] = tkinter.IntVar()
#         self.inParameters['Exposure'].set(10)
#         # noinspection PyTypeChecker
#         self.inParameters['Padding'] = tkinter.IntVar()
#         self.inParameters['Padding'].set(100)
#         # noinspection PyTypeChecker
#         self.inParameters['Width'] = tkinter.IntVar()
#         self.inParameters['Width'].set(1000)
#         # noinspection PyTypeChecker
#         self.inParameters['Height'] = tkinter.IntVar()
#         self.inParameters['Height'].set(1000)
#         self.imgSource = [0, 0, 0, 0]
#         self.imgSource_perspective = [0, 0, 0, 0]
#
#         self.padding = None
#         self.imgSource_calib = [
#             [[192.20404052734375, 188.24658203125],
#              [553.2216796875, 190.560791015625],
#              [551.6758422851562, 553.64501953125],
#              [190.57870483398438, 551.96728515625]],
#
#             [[427.5229797363281, 211.44573974609375],
#              [790.4320678710938, 212.46725463867188],
#              [790.6268920898438, 579.578369140625],
#              [428.3389892578125, 579.2514038085938]],
#
#             [[427.6001281738281, 741.3861694335938],
#              [786.5081176757812, 738.234130859375],
#              [789.7010498046875, 1099.710205078125],
#              [429.6021728515625, 1104.5653076171875]],
#
#             [[182.9971923828125, 772.1783447265625],
#              [546.138427734375, 764.6105346679688],
#              [553.9653930664062, 1129.499755859375],
#              [189.36074829101562, 1138.191650390625]]
#
#         ]
#
#         self.input_transform = [0, 0, 0, 0]
#         self.output_transform = [0, 0, 0, 0]
#         self.matrix = [0, 0, 0, 0]
#
#     def do(self):
#
#         start = time.time()
#         with Vimba.get_instance() as vim:
#             cams = vim.get_all_cameras()
#             for cam in cams:
#                 for cam_id in src.my_globals.CAMERA_LIST:
#                     if cam.get_serial() == cam_id:
#                         with cam as camera:
#                             camera.ExposureAuto.set('Off')
#                             camera.ExposureTimeAbs.set(self.inParameters['Exposure'].get() * 1000)
#                             frame = camera.get_frame()
#                             self.imgSource[src.my_globals.CAMERA_LIST.index(cam_id)] = numpy.concatenate(
#                                 frame.as_numpy_ndarray(), axis=1)
#                             h, w = self.imgSource[src.my_globals.CAMERA_LIST.index(cam_id)].shape[:2]
#                             newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
#                                 src.my_globals.k[src.my_globals.CAMERA_LIST.index(cam_id)],
#                                 src.my_globals.d[src.my_globals.CAMERA_LIST.index(cam_id)], (w, h), 1,
#                                 (w, h))
#                             dst = cv2.undistort(self.imgSource[src.my_globals.CAMERA_LIST.index(cam_id)],
#                                                 src.my_globals.k[src.my_globals.CAMERA_LIST.index(cam_id)],
#                                                 src.my_globals.d[src.my_globals.CAMERA_LIST.index(cam_id)],
#                                                 None,
#                                                 newcameramtx)
#                             x, y, w, h = roi
#                             dst = dst[y:y + h, x:x + w]
#                             self.imgSource[src.my_globals.CAMERA_LIST.index(cam_id)] = dst
#
#         pad = self.inParameters['Padding'].get()
#         width = self.inParameters['Width'].get() - 2 * pad
#         height = self.inParameters['Height'].get() - 2 * pad
#         # [left,right,top,bottom]
#         self.padding = [[0, 1, 0, 1],
#                         [1, 0, 0, 1],
#                         [1, 0, 1, 0],
#                         [0, 1, 1, 0]]
#         for n in range(4):
#             self.padding[n] = [pad * item for item in self.padding[n]]
#
#             self.input_transform[n] = numpy.float32(self.imgSource_calib[n])
#
#             # self.output_transform[n] = numpy.float32(dim)
#             self.output_transform[n] = numpy.float32(
#                 [[self.padding[n][0], self.padding[n][2]],
#                  [width + self.padding[n][0], self.padding[n][2]],
#                  [width + self.padding[n][0], height + self.padding[n][2]],
#                  [self.padding[n][0], height + self.padding[n][2]]])
#             # output_transform = numpy.float32(dim)
#
#             self.matrix[n] = cv2.getPerspectiveTransform(self.input_transform[n], self.output_transform[n])
#
#             self.imgSource_perspective[n] = cv2.warpPerspective(self.imgSource[n],
#                                                                 self.matrix[n],
#                                                                 (width + self.padding[n][0] + self.padding[n][1],
#                                                                  height + self.padding[n][2] + self.padding[n][3]),
#                                                                 cv2.INTER_NEAREST,
#                                                                 borderMode=cv2.BORDER_CONSTANT,
#                                                                 borderValue=(0, 0, 0))
#
#         # img12 = cv2.hconcat([self.imgSource[2], self.imgSource[3]])
#         # img34 = cv2.hconcat([self.imgSource[1], self.imgSource[0]])
#         img12 = cv2.hconcat([self.imgSource_perspective[2], self.imgSource_perspective[3]])
#         img34 = cv2.hconcat([self.imgSource_perspective[1], self.imgSource_perspective[0]])
#         self.img = cv2.vconcat([img12, img34])
#
#         self.img = cv2.flip(self.img, 0)
#         self.inParameters['height'] = self.img.shape[0]
#         self.inParameters['width'] = self.img.shape[1]
#         end = time.time()
#         print('time:', end - start)
#
#         # (int(self.inParameters['output.width'].get()
#         #      + self.inParameters['padding.left'].get()
#         #      + self.inParameters['padding.right'].get()),
#         #  int(self.inParameters['output.height'].get()
#         #      + self.inParameters['padding.bottom'].get()
#         #      + self.inParameters['padding.top'].get())),
#

# def canvas_img(image):
#     """
#
#     :param image:
#     :return:
#     """
#     save_img(image)
#     image = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(image))
#     return image
#
#
# def save_img(image):
#     filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     cv2.imwrite('img/' + filename + '.png', image)
#     # PIL.Image.fromarray(image).save('img/' + filename + '.png')
#
