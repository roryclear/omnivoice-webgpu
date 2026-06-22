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

    """Provide audio data from a sound file as NumPy array.

    By default, the whole file is read from the beginning, but the
    position to start reading can be specified with *start* and the
    number of frames to read can be specified with *frames*.
    Alternatively, a range can be specified with *start* and *stop*.

    If there is less data left in the file than requested, the rest of
    the frames are filled with *fill_value*.
    If no *fill_value* is specified, a smaller array is returned.

    Parameters
    ----------
    file : str or int or file-like object
        The file to read from.  See `SoundFile` for details.
    frames : int, optional
        The number of frames to read. If *frames* is negative, the whole
        rest of the file is read.  Not allowed if *stop* is given.
    start : int, optional
        Where to start reading.  A negative value counts from the end.
    stop : int, optional
        The index after the last frame to be read.  A negative value
        counts from the end.  Not allowed if *frames* is given.
    dtype : {'float64', 'float32', 'int32', 'int16'}, optional
        Data type of the returned array, by default ``'float64'``.
        Floating point audio data is typically in the range from
        ``-1.0`` to ``1.0``.  Integer data is in the range from
        ``-2**15`` to ``2**15-1`` for ``'int16'`` and from ``-2**31`` to
        ``2**31-1`` for ``'int32'``.

        .. note:: Reading int values from a float file will *not*
            scale the data to [-1.0, 1.0). If the file contains
            ``np.array([42.6], dtype='float32')``, you will read
            ``np.array([43], dtype='int32')`` for ``dtype='int32'``.

    Returns
    -------
    audiodata : `numpy.ndarray` or type(out)
        A two-dimensional (frames x channels) NumPy array is returned.
        If the sound file has only one channel, a one-dimensional array
        is returned.  Use ``always_2d=True`` to return a two-dimensional
        array anyway.

        If *out* was specified, it is returned.  If *out* has more
        frames than available in the file (or if *frames* is smaller
        than the length of *out*) and no *fill_value* is given, then
        only a part of *out* is overwritten and a view containing all
        valid frames is returned.
    samplerate : int
        The sample rate of the audio file.

    Other Parameters
    ----------------
    always_2d : bool, optional
        By default, reading a mono sound file will return a
        one-dimensional array.  With ``always_2d=True``, audio data is
        always returned as a two-dimensional array, even if the audio
        file has only one channel.
    fill_value : float, optional
        If more frames are requested than available in the file, the
        rest of the output is be filled with *fill_value*.  If
        *fill_value* is not specified, a smaller array is returned.
    out : `numpy.ndarray` or subclass, optional
        If *out* is specified, the data is written into the given array
        instead of creating a new array.  In this case, the arguments
        *dtype* and *always_2d* are silently ignored!  If *frames* is
        not given, it is obtained from the length of *out*.
    samplerate, channels, format, subtype, endian, closefd
        See `SoundFile`.

    Examples
    --------
    >>> import soundfile as sf
    >>> data, samplerate = sf.read('stereo_file.wav')
    >>> data
    array([[ 0.71329652,  0.06294799],
           [-0.26450912, -0.38874483],
           ...
           [ 0.67398441, -0.11516333]])
    >>> samplerate
    44100

    """
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
    """Write data to a sound file.

    .. note:: If *file* exists, it will be truncated and overwritten!

    Parameters
    ----------
    file : str or int or file-like object
        The file to write to.  See `SoundFile` for details.
    data : array_like
        The data to write.  Usually two-dimensional (frames x channels),
        but one-dimensional *data* can be used for mono files.
        Only the data types ``'float64'``, ``'float32'``, ``'int32'``
        and ``'int16'`` are supported.

        .. note:: The data type of *data* does **not** select the data
                  type of the written file. Audio data will be
                  converted to the given *subtype*. Writing int values
                  to a float file will *not* scale the values to
                  [-1.0, 1.0). If you write the value ``np.array([42],
                  dtype='int32')``, to a ``subtype='FLOAT'`` file, the
                  file will then contain ``np.array([42.],
                  dtype='float32')``.

    samplerate : int
        The sample rate of the audio data.
    subtype : str, optional
        See `default_subtype()` for the default value and
        `available_subtypes()` for all possible values.

    Other Parameters
    ----------------
    format, endian, closefd, compression_level, bitrate_mode
        See `SoundFile`.

    Examples
    --------
    Write 10 frames of random data to a new file:

    >>> import numpy as np
    >>> import soundfile as sf
    >>> sf.write('stereo_file.wav', np.random.randn(10, 2), 44100, 'PCM_24')

    """
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
    """A sound file.

    For more documentation see the __init__() docstring (which is also
    used for the online documentation (https://python-soundfile.readthedocs.io/).

    """

    def __init__(self, file: FileDescriptorOrPath, mode: str | None = 'r',
                 samplerate: int | None = None, channels: int | None = None,
                 subtype: str | None = None, endian: str | None = None,
                 format: str | None = None, closefd: bool = True,
                 compression_level: float | None = None,
                 bitrate_mode: str | None = None) -> None:
        """Open a sound file.

        If a file is opened with `mode` ``'r'`` (the default) or
        ``'r+'``, no sample rate, channels or file format need to be
        given because the information is obtained from the file. An
        exception is the ``'RAW'`` data format, which always requires
        these data points.

        File formats consist of three case-insensitive strings:

        * a *major format* which is by default obtained from the
          extension of the file name (if known) and which can be
          forced with the format argument (e.g. ``format='WAVEX'``).
        * a *subtype*, e.g. ``'PCM_24'``. Most major formats have a
          default subtype which is used if no subtype is specified.
        * an *endian-ness*, which doesn't have to be specified at all in
          most cases.

        A `SoundFile` object is a *context manager*, which means
        if used in a "with" statement, `close()` is automatically
        called when reaching the end of the code block inside the "with"
        statement.

        Parameters
        ----------
        file : str or int or file-like object
            The file to open.  This can be a file name, a file
            descriptor or a Python file object (or a similar object with
            the methods ``read()``/``readinto()``, ``write()``,
            ``seek()`` and ``tell()``).
        mode : {'r', 'r+', 'w', 'w+', 'x', 'x+'}, optional
            Open mode.  Has to begin with one of these three characters:
            ``'r'`` for reading, ``'w'`` for writing (truncates *file*)
            or ``'x'`` for writing (raises an error if *file* already
            exists).  Additionally, it may contain ``'+'`` to open
            *file* for both reading and writing.
            The character ``'b'`` for *binary mode* is implied because
            all sound files have to be opened in this mode.
            If *file* is a file descriptor or a file-like object,
            ``'w'`` doesn't truncate and ``'x'`` doesn't raise an error.
        samplerate : int
            The sample rate of the file.  If `mode` contains ``'r'``,
            this is obtained from the file (except for ``'RAW'`` files).
        channels : int
            The number of channels of the file.
            If `mode` contains ``'r'``, this is obtained from the file
            (except for ``'RAW'`` files).
        subtype : str, sometimes optional
            The subtype of the sound file.  If `mode` contains ``'r'``,
            this is obtained from the file (except for ``'RAW'``
            files), if not, the default value depends on the selected
            `format` (see `default_subtype()`).
            See `available_subtypes()` for all possible subtypes for
            a given `format`.
        endian : {'FILE', 'LITTLE', 'BIG', 'CPU'}, sometimes optional
            The endian-ness of the sound file.  If `mode` contains
            ``'r'``, this is obtained from the file (except for
            ``'RAW'`` files), if not, the default value is ``'FILE'``,
            which is correct in most cases.
        format : str, sometimes optional
            The major format of the sound file.  If `mode` contains
            ``'r'``, this is obtained from the file (except for
            ``'RAW'`` files), if not, the default value is determined
            from the file extension.  See `available_formats()` for
            all possible values.
        closefd : bool, optional
            Whether to close the file descriptor on `close()`. Only
            applicable if the *file* argument is a file descriptor.
        compression_level : float, optional
            The compression level on 'write()'. The compression level
            should be between 0.0 (minimum compression level) and 1.0
            (highest compression level).
            See `libsndfile document <https://github.com/libsndfile/libsndfile/blob/c81375f070f3c6764969a738eacded64f53a076e/docs/command.md>`__.
        bitrate_mode : {'CONSTANT', 'AVERAGE', 'VARIABLE'}, optional
            The bitrate mode on 'write()'.
            See `libsndfile document <https://github.com/libsndfile/libsndfile/blob/c81375f070f3c6764969a738eacded64f53a076e/docs/command.md>`__.

        Examples
        --------
        >>> from soundfile import SoundFile

        Open an existing file for reading:

        >>> myfile = SoundFile('existing_file.wav')
        >>> # do something with myfile
        >>> myfile.close()

        Create a new sound file for reading and writing using a with
        statement:

        >>> with SoundFile('new_file.wav', 'x+', 44100, 2) as myfile:
        >>>     # do something with myfile
        >>>     # ...
        >>>     assert not myfile.closed
        >>>     # myfile.close() is called automatically at the end
        >>> assert myfile.closed

        """
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
        if set(mode).issuperset('r+') and self.seekable():
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

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def seekable(self) -> bool:
        """Return True if the file supports seeking."""
        return self._info.seekable == _snd.SF_TRUE

    def seek(self, frames: int, whence: int = SEEK_SET) -> int:
        """Set the read/write position.

        Parameters
        ----------
        frames : int
            The frame index or offset to seek.
        whence : {SEEK_SET, SEEK_CUR, SEEK_END}, optional
            By default (``whence=SEEK_SET``), *frames* are counted from
            the beginning of the file.
            ``whence=SEEK_CUR`` seeks from the current position
            (positive and negative values are allowed for *frames*).
            ``whence=SEEK_END`` seeks from the end (use negative value
            for *frames*).

        Returns
        -------
        int
            The new absolute read/write position in frames.

        Examples
        --------
        >>> from soundfile import SoundFile, SEEK_END
        >>> myfile = SoundFile('stereo_file.wav')

        Seek to the beginning of the file:

        >>> myfile.seek(0)
        0

        Seek to the end of the file:

        >>> myfile.seek(0, SEEK_END)
        44100  # this is the file length

        """
        position = _snd.sf_seek(self._file, frames, whence)
        _error_check(self._errorcode)
        return position

    def tell(self) -> int:
        """Return the current read/write position."""
        return self.seek(0, SEEK_CUR)


    def read(self, frames, dtype):
   
        out = np.empty((frames, self.channels), dtype, order='C')
        frames = self._array_io('read', out, frames)

        return out

    def flush(self) -> None:
        """Write unwritten data to the file system.

        Data written with `write()` is not immediately written to
        the file system but buffered in memory to be written at a later
        time.  Calling `flush()` makes sure that all changes are
        actually written to the file system.

        This has no effect on files opened in read-only mode.

        """
        _snd.sf_write_sync(self._file)

    def close(self) -> None:
        """Close the file.  Can be called multiple times."""
        if not self.closed:
            # be sure to flush data to disk before closing the file
            self.flush()
            err = _snd.sf_close(self._file)
            self._file = None
            _error_check(err)

    # sf_error(NULL) returns a global (non-thread-safe) error code.
    # When an sf_open* call fails (returns NULL), we must call
    # sf_error(NULL) to retrieve the error code, but another thread
    # may clear the global error between our open and our sf_error.
    # This lock serialises the open+sf_error(NULL) pair to prevent that.
    _sf_error_lock = _threading.Lock()

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

        with self._sf_error_lock:
            file_ptr = openfunction(file, mode_int, self._info)
            if file_ptr == _ffi.NULL:
                err = _snd.sf_error(file_ptr)
                raise LibsndfileError(err, prefix=f"Error opening {self.name!r}: ")
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
        """Call one of libsndfile's read/write functions."""
        assert ctype in _ffi_types.values()
        curr = 0
        if self.seekable():
            curr = self.tell()
        func = getattr(_snd, 'sf_' + action + 'f_' + ctype)
        frames = func(self._file, data, frames)
        _error_check(self._errorcode)
        if self.seekable():
            self.seek(curr + frames, SEEK_SET)  # Update read & write position
        return frames

    def _prepare_read(self, start, stop, frames):
        """Seek to start frame and calculate length."""
        if start != 0 and not self.seekable():
            raise ValueError("start is only allowed for seekable files")
        if frames >= 0 and stop is not None:
            raise TypeError("Only one of {frames, stop} may be used")

        start, stop, _ = slice(start, stop).indices(self.frames)
        if stop < start:
            stop = start
        if frames < 0:
            frames = stop - start
        if self.seekable():
            self.seek(start, SEEK_SET)
        return frames

def _error_check(err, prefix=""):
    """Raise LibsndfileError if there is an error."""
    if err != 0:
        raise LibsndfileError(err, prefix=prefix)


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

