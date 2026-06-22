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
_ffi: Any

_str_types: Final[dict[str, int]] = {
    'title':       0x01,
    'copyright':   0x02,
    'software':    0x03,
    'artist':      0x04,
    'comment':     0x05,
    'date':        0x06,
    'album':       0x07,
    'license':     0x08,
    'tracknumber': 0x09,
    'genre':       0x10,
}

_formats: Final[dict[str, int]] = {
    'WAV':   0x010000,  # Microsoft WAV format (little endian default).
    'AIFF':  0x020000,  # Apple/SGI AIFF format (big endian).
    'AU':    0x030000,  # Sun/NeXT AU format (big endian).
    'RAW':   0x040000,  # RAW PCM data.
    'PAF':   0x050000,  # Ensoniq PARIS file format.
    'SVX':   0x060000,  # Amiga IFF / SVX8 / SV16 format.
    'NIST':  0x070000,  # Sphere NIST format.
    'VOC':   0x080000,  # VOC files.
    'IRCAM': 0x0A0000,  # Berkeley/IRCAM/CARL
    'W64':   0x0B0000,  # Sonic Foundry's 64 bit RIFF/WAV
    'MAT4':  0x0C0000,  # Matlab (tm) V4.2 / GNU Octave 2.0
    'MAT5':  0x0D0000,  # Matlab (tm) V5.0 / GNU Octave 2.1
    'PVF':   0x0E0000,  # Portable Voice Format
    'XI':    0x0F0000,  # Fasttracker 2 Extended Instrument
    'HTK':   0x100000,  # HMM Tool Kit format
    'SDS':   0x110000,  # Midi Sample Dump Standard
    'AVR':   0x120000,  # Audio Visual Research
    'WAVEX': 0x130000,  # MS WAVE with WAVEFORMATEX
    'SD2':   0x160000,  # Sound Designer 2
    'FLAC':  0x170000,  # FLAC lossless file format
    'CAF':   0x180000,  # Core Audio File format
    'WVE':   0x190000,  # Psion WVE format
    'OGG':   0x200000,  # Xiph OGG container
    'MPC2K': 0x210000,  # Akai MPC 2000 sampler
    'RF64':  0x220000,  # RF64 WAV file
    'MP3':   0x230000,  # MPEG-1/2 audio stream
}

_subtypes: Final[dict[str, int]] = {
    'PCM_S8':         0x0001,  # Signed 8 bit data
    'PCM_16':         0x0002,  # Signed 16 bit data
    'PCM_24':         0x0003,  # Signed 24 bit data
    'PCM_32':         0x0004,  # Signed 32 bit data
    'PCM_U8':         0x0005,  # Unsigned 8 bit data (WAV and RAW only)
    'FLOAT':          0x0006,  # 32 bit float data
    'DOUBLE':         0x0007,  # 64 bit float data
    'ULAW':           0x0010,  # U-Law encoded.
    'ALAW':           0x0011,  # A-Law encoded.
    'IMA_ADPCM':      0x0012,  # IMA ADPCM.
    'MS_ADPCM':       0x0013,  # Microsoft ADPCM.
    'GSM610':         0x0020,  # GSM 6.10 encoding.
    'VOX_ADPCM':      0x0021,  # OKI / Dialogix ADPCM
    'NMS_ADPCM_16':   0x0022,  # 16kbs NMS G721-variant encoding.
    'NMS_ADPCM_24':   0x0023,  # 24kbs NMS G721-variant encoding.
    'NMS_ADPCM_32':   0x0024,  # 32kbs NMS G721-variant encoding.
    'G721_32':        0x0030,  # 32kbs G721 ADPCM encoding.
    'G723_24':        0x0031,  # 24kbs G723 ADPCM encoding.
    'G723_40':        0x0032,  # 40kbs G723 ADPCM encoding.
    'DWVW_12':        0x0040,  # 12 bit Delta Width Variable Word encoding.
    'DWVW_16':        0x0041,  # 16 bit Delta Width Variable Word encoding.
    'DWVW_24':        0x0042,  # 24 bit Delta Width Variable Word encoding.
    'DWVW_N':         0x0043,  # N bit Delta Width Variable Word encoding.
    'DPCM_8':         0x0050,  # 8 bit differential PCM (XI only)
    'DPCM_16':        0x0051,  # 16 bit differential PCM (XI only)
    'VORBIS':         0x0060,  # Xiph Vorbis encoding.
    'OPUS':           0x0064,  # Xiph/Skype Opus encoding.
    'ALAC_16':        0x0070,  # Apple Lossless Audio Codec (16 bit).
    'ALAC_20':        0x0071,  # Apple Lossless Audio Codec (20 bit).
    'ALAC_24':        0x0072,  # Apple Lossless Audio Codec (24 bit).
    'ALAC_32':        0x0073,  # Apple Lossless Audio Codec (32 bit).
    'MPEG_LAYER_I':   0x0080,  # MPEG-1 Audio Layer I.
    'MPEG_LAYER_II':  0x0081,  # MPEG-1 Audio Layer II.
    'MPEG_LAYER_III': 0x0082,  # MPEG-2 Audio Layer III.
}

_endians: Final[dict[str, int]] = {
    'FILE':   0x00000000,  # Default file endian-ness.
    'LITTLE': 0x10000000,  # Force little endian-ness.
    'BIG':    0x20000000,  # Force big endian-ness.
    'CPU':    0x30000000,  # Force CPU endian-ness.
}

# libsndfile doesn't specify default subtypes, these are somehow arbitrary:
_default_subtypes: Final[dict[str, str]] = {
    'WAV':   'PCM_16',
    'AIFF':  'PCM_16',
    'AU':    'PCM_16',
    # 'RAW':  # subtype must be explicit!
    'PAF':   'PCM_16',
    'SVX':   'PCM_16',
    'NIST':  'PCM_16',
    'VOC':   'PCM_16',
    'IRCAM': 'PCM_16',
    'W64':   'PCM_16',
    'MAT4':  'DOUBLE',
    'MAT5':  'DOUBLE',
    'PVF':   'PCM_16',
    'XI':    'DPCM_16',
    'HTK':   'PCM_16',
    'SDS':   'PCM_16',
    'AVR':   'PCM_16',
    'WAVEX': 'PCM_16',
    'SD2':   'PCM_16',
    'FLAC':  'PCM_16',
    'CAF':   'PCM_16',
    'WVE':   'ALAW',
    'OGG':   'VORBIS',
    'MPC2K': 'PCM_16',
    'RF64':  'PCM_16',
    'MP3':   'MPEG_LAYER_III',
}

_ffi_types: Final[dict[str, str]] = {
    'float64': 'double',
    'float32': 'float',
    'int32': 'int',
    'int16': 'short'
}

_bitrate_modes: Final[dict[str, int]] = {
    'CONSTANT': 0,
    'AVERAGE': 1,
    'VARIABLE': 2,
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
        self._info = _create_info_struct(file, mode, samplerate, channels,
                                         format, subtype, endian)
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

    def _check_dtype(self, dtype):
        """Check if dtype string is valid and return ctype string."""
        try:
            return _ffi_types[dtype]
        except KeyError:
            raise ValueError(f"dtype must be one of {sorted(_ffi_types.keys())!r} and not {dtype!r}")

    def _array_io(self, action, array, frames):
        """Check array and call low-level IO function."""
        if array.ndim not in (1,2):
            raise ValueError(f"Invalid shape: {array.shape!r} ({'0 dimensions not supported' if array.ndim < 1 else 'too many dimensions'})")
        array_channels = 1 if array.ndim == 1 else array.shape[1]
        if array_channels != self.channels:
            raise ValueError(f"Invalid shape: {array.shape!r} (Expected {self.channels} channels, got {array_channels})")
        if not array.flags.c_contiguous:
            raise ValueError("Data must be C-contiguous")
        ctype = self._check_dtype(array.dtype.name)
        assert array.dtype.itemsize == _ffi.sizeof(ctype)
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


def _create_info_struct(file, mode, samplerate, channels,
                        format, subtype, endian):
    """Check arguments and create SF_INFO struct."""
    original_format = format
    if format is None:
        format = _get_format_from_filename(file, mode)
        assert isinstance(format, str)
    else:
        _check_format(format)

    info = _ffi.new("SF_INFO*")
    if 'r' not in mode or format.upper() == 'RAW':
        if samplerate is None:
            raise TypeError("samplerate must be specified")
        info.samplerate = samplerate
        if channels is None:
            raise TypeError("channels must be specified")
        info.channels = channels
        info.format = _format_int(format, subtype, endian)
    else:
        if any(arg is not None for arg in (
                samplerate, channels, original_format, subtype, endian)):
            raise TypeError("Not allowed for existing files (except 'RAW'): "
                            "samplerate, channels, format, subtype, endian")
    return info


def _get_format_from_filename(file, mode):
    """Return a format string obtained from file (or file.name).

    If file already exists (= read mode), an empty string is returned on
    error.  If not, an exception is raised.
    The return type will always be str or unicode (even if
    file/file.name is a bytes object).

    """
    format = ''
    file = getattr(file, 'name', file)
    try:
        # This raises an exception if file is not a (Unicode/byte) string:
        format = _os.path.splitext(file)[-1][1:]
        # Convert bytes to unicode (raises AttributeError on Python 3 str):
        format = format.decode('utf-8', 'replace')
    except Exception:
        pass
    if format.upper() not in _formats and 'r' not in mode:
        raise TypeError(f"No format specified and unable to get format from "
                        f"file extension: {file!r}")
    return format
