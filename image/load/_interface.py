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
    
    @abstractmethod
    def _load_image(self):
        ...

    @abstractmethod
    def _set_initial_loader_properties(self):
        ...

    @abstractmethod
    def _get_original_image_mode(self):
        ...

    @abstractmethod
    def _set_mode_description(self):
        ...

    @abstractmethod
    def _set_mode(self):
        ...

    @abstractmethod
    def _image_conversion_helper(self):
        ...
    
    @abstractmethod
    def _update_dtype(self):
        ...

    @abstractmethod
    def _set_image(self):
        ...
    
# -------------------------------------------------------------------------
