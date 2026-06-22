from mysoundfile import read
import struct
def load_waveform2(audio_path: str):
    data, sr = read(audio_path, dtype="float32", always_2d=True)
    return data.T, sr  # (T, C) → (C, T)

def load_waveform(audio_path: str):
    data, sr = read(audio_path, dtype="float32", always_2d=True)

    print(data.shape)

    bytes = open("voice.wav", "rb").read()
    sample_rate = struct.unpack_from('<I', bytes, 24)[0]
    print(sample_rate)

    return data.T, sr  # (T, C) → (C, T)


if __name__ == "__main__":
    x = load_waveform("voice.wav")