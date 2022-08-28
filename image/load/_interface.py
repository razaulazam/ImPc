# Copyright (C) Raza Ul Azam, All Rights Reserved.
# \brief Abstract base class for Image loader

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty

# -------------------------------------------------------------------------

class BaseImage(ABC):

    @abstractclassmethod
    def create_loader(cls):
        ...

    @abstractmethod
    def get_mode_description(self):
        ...

    @abstractproperty
    def image(self):
        ...

    @abstractproperty
    def height(self):
        ...

    @abstractproperty
    def width(self):
        ...

    @abstractproperty
    def dtype(self):
        ...

    @abstractproperty
    def dims(self):
        ...

    @abstractproperty
    def channels(self):
        ...

    @abstractproperty
    def mode(self):
        ...

    @abstractproperty
    def mode_description(self):
        ...

    @abstractmethod
    def is_rgb(self):
        ...

    @abstractmethod
    def is_gray(self):
        ...

    @abstractmethod
    def is_rgba(self):
        ...

    @abstractmethod
    def is_lab(self):
        ...

    @abstractmethod
    def is_hsv(self):
        ...

    @abstractmethod
    def is_ycbcr(self):
        ...

    @abstractmethod
    def normalize(self):
        ...

    @abstractmethod
    def copy(self):
        ...

    @abstractmethod
    def show(self):
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def putpixel(self):
        ...

    @abstractmethod
    def getpixel(self):
        ...

    @abstractmethod
    def tobytes(self):
        ...

    @abstractmethod
    def getbbox(self):
        ...

    @abstractmethod
    def close(self):
        ...
        
# -------------------------------------------------------------------------
