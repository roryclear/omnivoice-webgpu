from mysoundfile import read
import numpy as np
import struct
def load_waveform2(audio_path: str):
    data, sr = read(audio_path, dtype="float32", always_2d=True)
    return data.T, sr  # (T, C) → (C, T)

def load_waveform(audio_path: str):
    data2, sr = read(audio_path, dtype="float32", always_2d=True)
    print(data2.shape, data2)
    #return data.T, sr  # (T, C) → (C, T)

    data = open("voice.wav", "rb").read()
    sample_rate = struct.unpack_from('<I', data, 24)[0]
    channels = struct.unpack_from('<H', data, 22)[0]
    data_offset = data.find(b'data') + 8
    raw_samples = data[data_offset:]
    n_samples = len(raw_samples) // 2  # 2 bytes per int16
    samples = struct.unpack(f'<{n_samples}h', raw_samples)  # 'h' = int16
    audio = np.array(samples, dtype=np.float32).reshape(-1, channels)
    # Normalize to [-1.0, 1.0] (matching typical float32 WAV/libraries)
    audio /= 32768.0

    print(type(audio), type(data2))

    np.testing.assert_allclose(audio, data2)

    return audio.T, sample_rate



if __name__ == "__main__":
    x = load_waveform("voice.wav")