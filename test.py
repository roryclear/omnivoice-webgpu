from omni import OmniVoice
import soundfile as sf
import torch
from dataclasses import dataclass, fields

@dataclass
class OmniVoiceGenerationConfig:
    num_step: int = 32
    guidance_scale: float = 2.0
    t_shift: float = 0.1
    layer_penalty_factor: float = 5.0
    position_temperature: float = 5.0
    class_temperature: float = 0.0
    denoise: bool = True
    preprocess_prompt: bool = True
    postprocess_output: bool = True
    audio_chunk_duration: float = 15.0
    audio_chunk_threshold: float = 30.0

    @classmethod
    def from_dict(cls, kwargs_dict):
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs_dict.items() if k in valid_keys}
        return cls(**filtered)

if __name__ == "__main__":
  # Load the model
  model = OmniVoice.from_pretrained(
      "k2-fsa/OmniVoice",
      device_map="mps:0",
      dtype=torch.float16
  )

  # Generate audio

  audio = model.generate(
      text="Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? 谢谢你",
      ref_audio="voice.mp3",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
      num_inference_steps=256,
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  sf.write("out.wav", audio[0], 24000)
  
  audio = model.generate(
      text = "That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black",
      ref_audio="voice.mp3",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
      num_inference_steps=256,
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  sf.write("out_long.wav", audio[0], 24000)