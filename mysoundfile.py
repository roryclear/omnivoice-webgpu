# https://github.com/bastibe/python-soundfile/tree/6511532a7d65ff21caf1e7793e31c39f7ed95100 BSD-3-Clause license
"""python-soundfile is an audio library based on libsndfile, CFFI and NumPy.

Sound files can be read or written directly using the functions
`read()` and `write()`.
To read a sound file in a block-wise fashion, use `blocks()`.
Alternatively, sound files can be opened as `SoundFile` objects.

For further information, see https://python-soundfile.readthedocs.io/.

"""
__version__ = "0.14.0"

import os as _os
import sys as _sys
import threading as _threading
from collections.abc import Generator
from ctypes.util import find_library as _find_library
from os import SEEK_CUR, SEEK_END, SEEK_SET
from typing import Any, BinaryIO, Final, Literal, TypeAlias

import numpy
from typing_extensions import Self
import numpy as np

from _soundfile import ffi as _ffi

FileDescriptorOrPath: TypeAlias = str | int | BinaryIO | _os.PathLike[Any]
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
_path = _os.path.dirname(_soundfile_data.__file__)  # TypeError if __file__ is None
_full_path = _os.path.join(_path, _packaged_libname)
_snd = _ffi.dlopen(_full_path)  # OSError if file doesn't exist or can't be loaded


def read(file: FileDescriptorOrPath, frames: int = -1, start: int = 0, stop: int | None = None, dtype: dtype_str = 'float64',
        always_2d: bool = False, fill_value: float | None = None, out: AudioData | AudioData_2d | None = None,
        samplerate: int | None = None, channels: int | None = None, format: str | None = None, subtype: str | None = None,
        endian: str | None = None, closefd: bool = True) -> tuple[AudioData | AudioData_2d, int]:
    with SoundFile(file, 'r', samplerate, channels,
                   subtype, endian, format, closefd) as f:
        frames = f._prepare_read(start, stop, frames)
        data = f.read(frames, dtype)
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

        if isinstance(file, _os.PathLike):
            file = _os.fspath(file)
        self._name = file
        if mode is None:
            mode = getattr(file, 'mode', None)
            if mode is None:
                raise TypeError("Can not get `mode` from file. provided `mode` is None.") # Raises ValueError explicitly for type checking.
        mode_int = _check_mode(mode)
        self._mode = mode
        self._compression_level = compression_level
        self._bitrate_mode = bitrate_mode
        self._info = _ffi.new("SF_INFO*")
        self._file = self._open(file, mode_int, closefd)
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
    """The number of channels in the sound file."""
    format = property(
        lambda self: _format_str(self._info.format & _snd.SF_FORMAT_TYPEMASK))
    """The major format of the sound file."""
    subtype = property(
        lambda self: _format_str(self._info.format & _snd.SF_FORMAT_SUBMASK))
    """The subtype of data in the the sound file."""
    endian = property(
        lambda self: _format_str(self._info.format & _snd.SF_FORMAT_ENDMASK))
    """The endian-ness of the data in the sound file."""
    format_info = property(
        lambda self: _format_info(self._info.format &
                                  _snd.SF_FORMAT_TYPEMASK)[1])
    """A description of the major format of the sound file."""
    subtype_info = property(
        lambda self: _format_info(self._info.format &
                                  _snd.SF_FORMAT_SUBMASK)[1])
    """A description of the subtype of the sound file."""
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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None: return

    def seek(self, frames: int, whence: int = SEEK_SET) -> int:
        position = _snd.sf_seek(self._file, frames, whence)
        return position

    def tell(self) -> int:
        """Return the current read/write position."""
        return self.seek(0, SEEK_CUR)

    def read(self, frames, dtype):
        out = np.empty((frames, self.channels), dtype, order='C')
        frames = self._array_io('read', out, frames)
        return out

    def _open(self, file, mode_int, closefd):
        """Call the appropriate sf_open*() function from libsndfile."""
        if isinstance(file, (str, bytes)):
            if _os.path.isfile(file):
                if 'x' in self.mode:
                    raise OSError(f"File exists: {self.name!r}")
                elif set(self.mode).issuperset('w+'):
                    # truncate the file, because SFM_RDWR doesn't:
                    _os.close(_os.open(file, _os.O_WRONLY | _os.O_TRUNC))
            openfunction = _snd.sf_open
            if isinstance(file, str):
                if _sys.platform == 'win32':
                    openfunction = _snd.sf_wchar_open
                else:
                    file = file.encode(_sys.getfilesystemencoding())
        elif isinstance(file, int):
            openfunction = lambda file, mode_int, info: _snd.sf_open_fd(file, mode_int, info, closefd)
        elif _has_virtual_io_attrs(file, mode_int):
            openfunction = lambda file, mode_int, info: _snd.sf_open_virtual(self._init_virtual_io(file),
                                            mode_int, info, _ffi.NULL)
        else:
            raise TypeError(f"Invalid file: {self.name!r}")

        file_ptr = openfunction(file, mode_int, self._info)
        if mode_int == _snd.SFM_WRITE:
            # Due to a bug in libsndfile version <= 1.0.25, frames != 0
            # when opening a named pipe in SFM_WRITE mode.
            # See http://github.com/erikd/libsndfile/issues/77.
            self._info.frames = 0
            # This is not necessary for "normal" files (because
            # frames == 0 in this case), but it doesn't hurt, either.
        return file_ptr

    def _array_io(self, action, array, frames):
        ctype = _ffi_types[array.dtype.name]
        cdata = _ffi.cast(ctype + '*', array.__array_interface__['data'][0])
        return self._cdata_io(action, cdata, ctype, frames)

    def _cdata_io(self, action, data, ctype, frames):
        curr = 0
        curr = self.tell()
        func = getattr(_snd, 'sf_' + action + 'f_' + ctype)
        frames = func(self._file, data, frames)
        self.seek(curr + frames, SEEK_SET)  # Update read & write position
        return frames

    def _prepare_read(self, start, stop, frames):
        start, stop, _ = slice(start, stop).indices(self.frames)
        if stop < start:
            stop = start
        if frames < 0:
            frames = stop - start
        self.seek(start, SEEK_SET)
        return frames

def _check_mode(mode):
    """Check if mode is valid and return its integer representation."""
    if not isinstance(mode, str):
        raise TypeError(f"Invalid mode: {mode!r}")
    mode_set = set(mode)
    if mode_set.difference('xrwb+') or len(mode) > len(mode_set):
        raise ValueError(f"Invalid mode: {mode!r}")
    if len(mode_set.intersection('xrw')) != 1:
        raise ValueError("mode must contain exactly one of 'xrw'")

    if '+' in mode_set:
        mode_int = _snd.SFM_RDWR
    elif 'r' in mode_set:
        mode_int = _snd.SFM_READ
    else:
        mode_int = _snd.SFM_WRITE
    return mode_int

