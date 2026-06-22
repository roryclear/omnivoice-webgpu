# https://github.com/bastibe/python-soundfile/tree/6511532a7d65ff21caf1e7793e31c39f7ed95100 BSD-3-Clause license
import os
import sys as _sys
from os import SEEK_SET
from typing import Any, BinaryIO, Final, Literal, TypeAlias

import numpy
from typing_extensions import Self
import numpy as np

from _soundfile import ffi as _ffi

FileDescriptorOrPath: TypeAlias = str | int | BinaryIO | os.PathLike[Any]
AudioData: TypeAlias = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.float32 | numpy.float64 | numpy.int32 | numpy.int16]]
AudioData_2d: TypeAlias = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float32 | numpy.float64 | numpy.int32 | numpy.int16]]
dtype_str: TypeAlias = Literal['float64', 'float32', 'int32', 'int16']
_snd: Any


_ffi_types: Final[dict[str, str]] = {
    'float64': 'double',
    'float32': 'float',
    'int32': 'int',
    'int16': 'short'
}

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
        frames = f._prepare_read(start, stop, frames)

        data = np.empty((frames, f.channels), dtype, order='C')
        f._array_io('read', data, frames)
    return data, f.samplerate



def write(file: FileDescriptorOrPath, data: AudioData, samplerate: int,
          subtype: str | None = None, endian: str | None = None,
          format: str | None = None, closefd: bool = True,
          compression_level: float | None = None,
          bitrate_mode: str | None = None) -> None:

    import numpy as np
    data = np.asarray(data)
    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[1]
    with SoundFile(file, 'w', samplerate, channels,
                   subtype, endian, format, closefd,
                   compression_level, bitrate_mode) as f: f.write(data)

class SoundFile:
    def __init__(self, file: FileDescriptorOrPath, mode: str | None = 'r',
                 samplerate: int | None = None, channels: int | None = None,
                 subtype: str | None = None, endian: str | None = None,
                 format: str | None = None, closefd: bool = True,
                 compression_level: float | None = None,
                 bitrate_mode: str | None = None) -> None:

        if isinstance(file, os.PathLike):
            file = os.fspath(file)
        self._name = file
        if mode is None:
            mode = getattr(file, 'mode', None)
            if mode is None:
                raise TypeError("Can not get `mode` from file. provided `mode` is None.") # Raises ValueError explicitly for type checking.
        self._mode = mode
        self._compression_level = compression_level
        self._bitrate_mode = bitrate_mode
        self._info = _ffi.new("SF_INFO*")
        self._file = self._open(file)
        if set(mode).issuperset('r+'):
            # Move write position to 0 (like in Python file objects)
            self.seek(0)
        _snd.sf_command(self._file, _snd.SFC_SET_CLIPPING, _ffi.NULL,
                        _snd.SF_TRUE)

        # set compression setting
        if self._compression_level is not None:
            # needs to be called before set_bitrate_mode
            self._set_compression_level(self._compression_level)
            if self._bitrate_mode is not None:
                self._set_bitrate_mode(self._bitrate_mode)

    name = property(lambda self: self._name)
    """The file name of the sound file."""
    mode = property(lambda self: self._mode)
    """The open mode the sound file was opened with."""
    samplerate = property(lambda self: self._info.samplerate)
    """The sample rate of the sound file."""
    frames = property(lambda self: self._info.frames)
    """The number of frames in the sound file."""
    channels = property(lambda self: self._info.channels)
    sections = property(lambda self: self._info.sections)
    """The number of sections of the sound file."""
    closed = property(lambda self: self._file is None)
    """Whether the sound file is closed or not."""
    _errorcode = property(lambda self: _snd.sf_error(self._file))
    """A pending sndfile error code."""
    compression_level = property(lambda self: self._compression_level)
    """The compression level on 'write()'"""
    bitrate_mode = property(lambda self: self._bitrate_mode)
    """The bitrate mode on 'write()'"""

    @property
    def extra_info(self):
        """Retrieve the log string generated when opening the file."""
        info = _ffi.new("char[]", 2**14)
        _snd.sf_command(self._file, _snd.SFC_GET_LOG_INFO,
                        info, _ffi.sizeof(info))
        return _ffi.string(info).decode('utf-8', 'replace')

    # avoid confusion if something goes wrong before assigning self._file:
    _file = None

    def __repr__(self) -> str:
        compression_setting = (f", compression_level={self.compression_level}"
                               if self.compression_level is not None else "")
        compression_setting += (f", bitrate_mode='{self.bitrate_mode}'"
                                if self.bitrate_mode is not None else "")
        return (f"SoundFile({self.name!r}, mode={self.mode!r}, "
                f"samplerate={self.samplerate}, channels={self.channels}, "
                f"format={self.format!r}, subtype={self.subtype!r}, "
                f"endian={self.endian!r}{compression_setting})")

    def __del__(self) -> None: return

    def __enter__(self) -> Self: return self

    def __exit__(self, *args: Any) -> None: return

    def seek(self, frames: int, whence: int = SEEK_SET) -> int: return _snd.sf_seek(self._file, frames, whence)

    def _open(self, file):
        file = file.encode(_sys.getfilesystemencoding())
        return _snd.sf_open(file, 16, self._info)

    def _array_io(self, action, array, frames):
        ctype = _ffi_types[array.dtype.name]
        cdata = _ffi.cast(ctype + '*', array.__array_interface__['data'][0])
        return self._cdata_io(action, cdata, ctype, frames)

    def _cdata_io(self, action, data, ctype, frames):
        func = getattr(_snd, 'sf_' + action + 'f_' + ctype)
        frames = func(self._file, data, frames)
        return frames

    def _prepare_read(self, start, stop, frames):
        start, stop, _ = slice(start, stop).indices(self.frames)
        return stop - start
