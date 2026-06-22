from omni import OmniVoice
import soundfile as sf
import torch
torch.manual_seed(4)
import pickle
import numpy as np

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
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  #pickle.dump(audio, open("short.pkl", "wb"))
  exp = pickle.load(open("short.pkl", "rb"))
  sf.write("out.wav", audio, 24000)
  np.testing.assert_allclose(exp, audio, rtol=1e-5)
  
  audio = model.generate(
      text = "That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black. That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black",
      ref_audio="voice.mp3",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  #pickle.dump(audio, open("long.pkl", "wb"))
  exp = pickle.load(open("long.pkl", "rb"))
  sf.write("out_long.wav", audio, 24000)
  np.testing.assert_allclose(exp, audio, rtol=1e-5)
  

  audio = model.generate(
      text="Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? 谢谢你",
      ref_audio="voice2.wav",
      ref_text="And eh all of the people, I mean we have the greatest military anywhere in the world, and you saw that, in Iran, where, in one week virtually, we knocked out their entire navy, their entire air force",
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  sf.write("out2.wav", audio, 24000)
