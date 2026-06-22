# https://github.com/bastibe/python-soundfile/tree/6511532a7d65ff21caf1e7793e31c39f7ed95100 BSD-3-Clause license
import os
import sys as _sys
from os import SEEK_SET
from typing import Any, BinaryIO, Final, Literal, TypeAlias

import numpy
import numpy as np

from _soundfile import ffi as _ffi

FileDescriptorOrPath: TypeAlias = str | int | BinaryIO | os.PathLike[Any]
AudioData: TypeAlias = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.float32 | numpy.float64 | numpy.int32 | numpy.int16]]
AudioData_2d: TypeAlias = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float32 | numpy.float64 | numpy.int32 | numpy.int16]]
dtype_str: TypeAlias = Literal['float64', 'float32', 'int32', 'int16']
_snd: Any

if _sys.platform == 'darwin':
    from platform import machine as _machine
    _packaged_libname = 'libsndfile_' + _machine() + '.dylib'


import _soundfile_data  # ImportError if this doesn't exist
_path = os.path.dirname(_soundfile_data.__file__)  # TypeError if __file__ is None
_full_path = os.path.join(_path, _packaged_libname)
_snd = _ffi.dlopen(_full_path)  # OSError if file doesn't exist or can't be loaded


def read(file: FileDescriptorOrPath, frames: int = -1, start: int = 0, stop: int | None = None, dtype: dtype_str = 'float64',
        always_2d: bool = False, fill_value: float | None = None, out: AudioData | AudioData_2d | None = None,
        samplerate: int | None = None, channels: int | None = None, format: str | None = None, subtype: str | None = None,
        endian: str | None = None, closefd: bool = True) -> tuple[AudioData | AudioData_2d, int]:
    with SoundFile(file, 'r', samplerate, channels,
                   subtype, endian, format, closefd) as f:
        data = np.empty((f.frames, f.channels), dtype, order='C')
        f._array_io('read', data, f.frames)
    return data, f.samplerate

class SoundFile:
    def __init__(self, file: FileDescriptorOrPath, mode: str | None = 'r',
                 samplerate: int | None = None, channels: int | None = None,
                 subtype: str | None = None, endian: str | None = None,
                 format: str | None = None, closefd: bool = True,
                 compression_level: float | None = None,
                 bitrate_mode: str | None = None) -> None:

        if isinstance(file, os.PathLike):
            file = os.fspath(file)
        self._compression_level = compression_level
        self._bitrate_mode = bitrate_mode
        self._info = _ffi.new("SF_INFO*")
        self._file = self._open(file)
        _snd.sf_command(self._file, _snd.SFC_SET_CLIPPING, _ffi.NULL,
                        _snd.SF_TRUE)


    samplerate = property(lambda self: self._info.samplerate)
    """The sample rate of the sound file."""
    frames = property(lambda self: self._info.frames)
    """The number of frames in the sound file."""
    channels = property(lambda self: self._info.channels)

    def __repr__(self): return
    def __del__(self): return
    def __enter__(self): return self
    def __exit__(self, *args): return

    def _open(self, file):
        file = file.encode(_sys.getfilesystemencoding())
        return _snd.sf_open(file, 16, self._info)

    def _array_io(self, action, array, frames):
        cdata = _ffi.cast("float" + '*', array.__array_interface__['data'][0])
        self._cdata_io(action, cdata, "float", frames)

    def _cdata_io(self, action, data, ctype, frames):
        func = getattr(_snd, 'sf_' + action + 'f_' + ctype)
        func(self._file, data, frames)

