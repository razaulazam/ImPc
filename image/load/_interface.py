# Copyright (C) 2022 FARO Technologies Inc., All Rights Reserved.
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
    def file_stream(self):
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
