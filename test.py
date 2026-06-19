from omnivoice import OmniVoice
import soundfile as sf
import torch
from typing import Union, Optional
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

# https://github.com/k2-fsa/OmniVoice/blob/9948396864cb713b0c2f92495cf4449bd8717127/omnivoice/models/omnivoice.py#L475
@torch.inference_mode()
def generate(
    model,
    text: Union[str, list[str]],
    language: Union[str, list[str], None] = None,
    ref_text: Union[str, list[str], None] = None,
    ref_audio: Union[
        str,
        list[str],
        tuple[torch.Tensor, int],
        list[tuple[torch.Tensor, int]],
        None,
    ] = None,
    voice_clone_prompt = None,
    instruct: Union[str, list[str], None] = None,
    duration: Union[float, list[Optional[float]], None] = None,
    speed: Union[float, list[Optional[float]], None] = None,
    generation_config=None,
    **kwargs,
) -> list:
    if model.audio_tokenizer is None or model.text_tokenizer is None:
        raise RuntimeError(
            "Model is not loaded with audio/text tokenizers. Make sure you "
            "loaded the model with OmniVoice.from_pretrained()."
        )
    
    gen_config = (
        generation_config
        if generation_config is not None
        else OmniVoiceGenerationConfig.from_dict(kwargs)
    )

    model.eval()

    full_task = model._preprocess_all(
        text=text,
        language=language,
        ref_text=ref_text,
        ref_audio=ref_audio,
        voice_clone_prompt=voice_clone_prompt,
        instruct=instruct,
        preprocess_prompt=gen_config.preprocess_prompt,
        speed=speed,
        duration=duration,
    )

    short_idx, long_idx = full_task.get_indices(
        gen_config, model.audio_tokenizer.config.frame_rate
    )

    results = [None] * full_task.batch_size

    if short_idx:
        short_task = full_task.slice_task(short_idx)
        short_results = model._generate_iterative(short_task, gen_config)
        for idx, res in zip(short_idx, short_results):
            results[idx] = res

    if long_idx:
        long_task = full_task.slice_task(long_idx)
        long_results = model._generate_chunked(long_task, gen_config)
        for idx, res in zip(long_idx, long_results):
            results[idx] = res

    generated_audios = []
    for i in range(full_task.batch_size):
        assert results[i] is not None, f"Result {i} was not generated"
        generated_audios.append(
            model._decode_and_post_process(
                results[i], full_task.ref_rms[i], gen_config  # type: ignore[arg-type]
            )
        )

    return generated_audios

if __name__ == "__main__":
  # Load the model
  model = OmniVoice.from_pretrained(
      "k2-fsa/OmniVoice",
      device_map="mps:0",
      dtype=torch.float16
  )

  # Generate audio
  audio = generate(model,
      text="Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? 谢谢你",
      ref_audio="voice.mp3",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
      num_inference_steps=256,
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.

  sf.write("out.wav", audio[0], 24000)