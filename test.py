from omnivoice import OmniVoice
import soundfile as sf
import torch

# Load the model
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="mps:0",
    dtype=torch.float16
)

# Generate audio
audio = model.generate(
    text="Testing testing one two three, this is made with Omni-Voice. Can you hear me or not? 谢谢你",
    ref_audio="voice.mp3",
    ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
    num_inference_steps=256,
) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.

sf.write("out.wav", audio[0], 24000)