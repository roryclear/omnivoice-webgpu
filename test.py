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
      text = "That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black, the beaming heat on their faces Then a figure emerges from the wastage Eyes transfixed with a piercing gaze One hand clutching his sword, raised to the sky They wonder how, they wonder why The sky turns white, it all becomes clear They felt lifted from their fears They shed tears in the light after six dark years Young bold soldiers, the fire burns, cracks and smoulders Five years older and wiser The fires are burning, on fire, never tire Slay warriors in the forests and on higher, we sing Hear the strings rising, the war's over, the bells ring Memories fading, soldiers slaying, looks like geezers raving The hazy fog over the Bullring, the lazy ways the birds sing A new baby's born every day, few men may be scorned today But look at things the other way 'Cause it may well be your final day And then the crowds roar, they slay, they all say I produced this using only my bare wit Give me a jungle or garage beat and admit defeat Use war and past injury as my metaphor and simile Get all applications in to me before the deadline 'Cause it's a fine line between strifeful crimes and a life of crime But you will reach the day And it's all mine, you can take it or leave it I shake and reveal stage tricks like Jimi Hendrix In the afterlife, gladiators meet their maker Float through the wheat fields and lakes of blue water To the next life from the fortress Away from the knives and slaughter To their wives and daughters Once more before the Lord judges over all of us It's in this place you'll see me Brace yourself, 'cause this goes deep I'll show you the secrets, the sky and the birds Actions speak louder than words Stand by me, my apprentice Be brave, clench fists",
      ref_audio="voice.mp3",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
      num_inference_steps=256,
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  sf.write("out_long.wav", audio[0], 24000)