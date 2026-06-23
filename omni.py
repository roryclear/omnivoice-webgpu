import logging
import math
import os
import re
import struct
from typing import List, Optional, Union
import pickle

import numpy as np
import torch
import torchaudio
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModel,
    HiggsAudioV2TokenizerModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING, AutoConfig

from tinygrad.helpers import partition
import json, urllib.request, typing, unicodedata, sys

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id'))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'kimi-k2': return self.encode("<|im_" + role + "|>" + role + "<|im_middle|>")
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'glm4': return self.encode("<|" + role + "|>")
    if self.preset == 'tekken':
      if role == 'user': return self.encode("[INST]")
      if role == 'assistant': return []
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.preset}'")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'kimi-k2': return [self.eos_id]
    if self.preset == 'qwen2': return [self.eos_id] + self.encode("\n")
    if self.preset == 'glm4': return []
    if self.preset == 'tekken': return self.encode("[/INST]")
    return [self.eos_id]
  def prefix(self) -> list[int]:
    return ([] if self.bos_id is None else [self.bos_id]) + (self.encode("<sop>") if self.preset == 'glm4' else [])
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id)

# https://github.com/huggingface/transformers/blob/1c75d06e73bf25d48a4379b9452ca009da9cf0a1/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py#L41
def encode(tok, input_values: torch.Tensor) -> torch.Tensor:
    e_semantic_input = _extract_semantic_features(tok, input_values).detach()
    e_semantic = semantic_encode(tok.encoder_semantic, e_semantic_input.transpose(1, 2))
    e_acoustic = acoustic_encode(tok.acoustic_encoder, input_values)
    embeddings = torch.cat([e_acoustic.to(e_semantic.device), e_semantic], dim=1)
    embeddings = tok.fc(embeddings.transpose(1, 2)).transpose(1, 2)
    audio_codes = quantizer_encode(tok.quantizer, embeddings)
    audio_codes = audio_codes.transpose(0, 1)
    return audio_codes
   
def quantizer_encode(quantizer, embeddings: torch.Tensor) -> torch.Tensor:
    residual = embeddings
    all_indices = []
    for q in quantizer.quantizers:
        hidden_states = residual.permute(0, 2, 1)
        hidden_states = q.project_in(hidden_states)

        shape = hidden_states.shape
        hidden_states = hidden_states.reshape((-1, shape[-1]))

        embed = q.codebook.embed.t()
        scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
        dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
        indices = dist.max(dim=-1).indices

        indices = indices.view(*shape[:-1])
        quantized = F.embedding(indices, q.codebook.embed)
        quantized = q.project_out(quantized)
        quantized = quantized.permute(0, 2, 1)

        residual = residual - quantized
        all_indices.append(indices)
    out_indices = torch.stack(all_indices)
    return out_indices

def semantic_encode(encoder, hidden_state: torch.Tensor) -> torch.Tensor:
    hidden_state = encoder.conv(hidden_state)
    for block in encoder.conv_blocks:
        hidden_state = block(hidden_state)
    return hidden_state

def acoustic_encode(encoder, hidden_state):
    hidden_state = encoder.conv1(hidden_state)
    for module in encoder.block:
        hidden_state = module(hidden_state)
    hidden_state = encoder.snake1(hidden_state)
    return encoder.conv2(hidden_state)

def _extract_semantic_features(tok, input_values: torch.FloatTensor) -> torch.FloatTensor:
    input_values = input_values[:, 0, :]
    # TODO: there is a diff here with original codebase https://github.com/boson-ai/higgs-audio/blob/f644b62b855ba2b938896436221e01efadcc76ca/boson_multimodal/audio_processing/higgs_audio_v2_tokenizer.py#L173-L174
    # input_values = F.pad(input_values, (self.pad, self.pad))
    input_values = F.pad(input_values, (160, 160))
    with torch.no_grad():
        outputs = tok.semantic_model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    stacked = torch.stack([h.to(input_values.device) for h in hidden_states], dim=1)
    semantic_features = stacked.mean(dim=1)
    semantic_features = semantic_features[:, :: tok.config.semantic_downsample_factor, :]
    return semantic_features

FRAME_RATE = 25
AUDIO_CHUNK_DURATION = 15.0
NUM_STEPS = 32
POSITION_TEMP = 5.0
LAYER_PENTALTY_FACTOR = 5.0
GUIDANCE_SCALE = 2.0
T_SHIFT = 0.1
AUDIO_CHUNKED_THRESHOLD = 30.0

class OmniVoiceConfig(PretrainedConfig):
    model_type = "omnivoice"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        llm_config: Optional[Union[dict, PretrainedConfig]] = None,
        **kwargs,
    ):

        if isinstance(llm_config, dict):
            self.llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)


        super().__init__(**kwargs)


def _resolve_model_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)

class blank: pass

HIDDEN_SIZE = 1024
NUM_AUDIO_CODEBOOK = 8
AUDIO_VOCAB_SIZE = 1025
AUDIO_CODEBOOK_WEIGHTS = [8, 8, 6, 6, 4, 4, 2, 2]
AUDIO_MASK_ID = 1024
SAMPLING_RATE = 24000
# saved from getting all chars with https://github.com/k2-fsa/OmniVoice/blob/9948396864cb713b0c2f92495cf4449bd8717127/omnivoice/utils/duration.py#L204
CHAR_WEIGHTS = pickle.load(open('char_weights.pkl', 'rb'))

data = json.load(urllib.request.urlopen("https://huggingface.co/k2-fsa/OmniVoice/resolve/main/tokenizer.json"))
special_tokens = data["added_tokens"]
special_tokens = {item['content']: item['id'] for item in special_tokens}
tok = SimpleTokenizer(normal_tokens=data["model"]["vocab"], special_tokens=special_tokens)

def load_waveform(audio_path: str):
    data = open(audio_path, "rb").read()
    sample_rate = struct.unpack_from('<I', data, 24)[0]
    channels = struct.unpack_from('<H', data, 22)[0]
    data_offset = data.find(b'data') + 8
    raw_samples = data[data_offset:]
    n_samples = len(raw_samples) // 2  # 2 bytes per int16
    samples = struct.unpack(f'<{n_samples}h', raw_samples)  # 'h' = int16
    audio = np.array(samples, dtype=np.float32).reshape(-1, channels)
    # Normalize to [-1.0, 1.0] (matching typical float32 WAV/libraries)
    audio /= 32768.0
    return audio.T, sample_rate

def load_audio(audio_path: str, sampling_rate: int) -> np.ndarray:
    data, sr = load_waveform(audio_path)

    # two sides
    data = np.mean(data, axis=0, keepdims=True)
    # just resample every time?
    data = torchaudio.functional.resample(torch.from_numpy(data), orig_freq=sr, new_freq=sampling_rate).numpy()
    return data

import inspect
class OmniVoice(PreTrainedModel):
    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class = OmniVoiceConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.llm = AutoModel.from_config(self.config.llm_config)

        self.audio_embeddings = nn.Embedding(NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, HIDDEN_SIZE)
        self.audio_heads = nn.Linear(HIDDEN_SIZE, NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, bias=False)

        # todo, breaks without this
        self.all_tied_weights_keys = {}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        logging.disable(logging.INFO)

        # Resolve to local path first; download only if not already cached
        resolved_path = _resolve_model_path(pretrained_model_name_or_path)
        model = super().from_pretrained(resolved_path, *args, **kwargs)

        audio_tokenizer_path = os.path.join(resolved_path, "audio_tokenizer")

        # higgs-audio-v2-tokenizer does not support MPS
        # (output channels > 65536)
        model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(audio_tokenizer_path, device_map="mps")
        model.audio_tokenizer.config.semantic_sample_rate = SAMPLING_RATE

        return model

def decode(decoder, audio_codes: torch.Tensor,):
    audio_codes = audio_codes.transpose(0, 1)
    quantized_out = torch.tensor(0.0)
    for i, indices in enumerate(audio_codes):
        quantizer = decoder.quantizer.quantizers[i]
        quantized = F.embedding(indices, quantizer.codebook.embed)
        quantized = quantizer.project_out(quantized)
        quantized = quantized.permute(0, 2, 1)
        quantized_out = quantized_out + quantized
    quantized = quantized_out
    quantized_acoustic = decoder.fc2(quantized.transpose(1, 2)).transpose(1, 2)
    hidden_state = decoder.acoustic_decoder.conv1(quantized_acoustic)

    for layer in decoder.acoustic_decoder.block:
        hidden_state = layer(hidden_state)

    hidden_state = decoder.acoustic_decoder.snake1(hidden_state)
    hidden_state = decoder.acoustic_decoder.conv2(hidden_state)
    hidden_state = decoder.acoustic_decoder.tanh(hidden_state)

    return hidden_state

def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise

_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)

class tiny_omni:
  def __init__(self, model):
    self.audio_tokenizer = model.audio_tokenizer
    self.device = model.device
    self.llm = model.llm
    self.codebook_layer_offsets = (torch.arange(NUM_AUDIO_CODEBOOK) * AUDIO_VOCAB_SIZE).to(self.device)
    self.audio_embeddings = nn.Embedding(NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, HIDDEN_SIZE)
    self.audio_embeddings.weight = model.audio_embeddings.weight
    self.audio_heads = nn.Linear(HIDDEN_SIZE, NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, bias=False)
    self.audio_heads.weight = model.audio_heads.weight

  def _prepare_embed_inputs(
      self, input_ids: torch.Tensor, audio_mask: torch.Tensor
  ) -> torch.Tensor:
      """
      Prepares embeddings from input_ids of shape (batch_size, layers, seq_length).
      Embedding shape is (batch_size, seq_length, hidden_size).
      """
      text_embeds = self.llm.get_input_embeddings()(input_ids[:, 0, :])

      # Apply shift to audio IDs based on codebook layer
      # audio_ids: [Batch, 8, Seq]
      # codebook_layer_offsets: [1, 8, 1]
      # Result: Layer 0 ID Layer 1 ID + Layer 2 ID + 2050...
      shifted_ids = (
          input_ids * audio_mask.unsqueeze(1)
      ) + self.codebook_layer_offsets.view(1, -1, 1)

      # input: [Batch, 8, Seq] -> output: [Batch, Seq, Hidden]
      audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)
      return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

  def __call__(
      self,
      input_ids: torch.LongTensor,
      audio_mask: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
  ):

      inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)
      

      llm_outputs = self.llm(
          inputs_embeds=inputs_embeds,
          attention_mask=attention_mask,
          position_ids=position_ids,
      )
      hidden_states = llm_outputs[0]

      # Shape: [B, S, C * Vocab]
      batch_size, seq_len, _ = hidden_states.shape
      logits_flat = self.audio_heads(hidden_states)
      # Shape: [B, S, C, Vocab] -> [B, C, S, Vocab]
      audio_logits = logits_flat.view(
          batch_size,
          seq_len,
          NUM_AUDIO_CODEBOOK,
          AUDIO_VOCAB_SIZE,
      ).permute(0, 2, 1, 3)
      return audio_logits

  @torch.inference_mode()
  def generate(
      self,
      text=None,
      ref_text=None,
      ref_audio=None,
  ) -> list[np.ndarray]:
      ref_audio_tokens = self.create_voice_clone_prompt(ref_audio=ref_audio)
      num_target_tokens = self._estimate_target_tokens(text, ref_text, ref_audio_tokens.size(-1),)

      result = self._generate_chunked(target_length=num_target_tokens, text=text,\
          ref_text=ref_text, ref_audio_tokens=ref_audio_tokens) 
      return self._decode_and_post_process(result)    

  def create_voice_clone_prompt(self, ref_audio: Union[str, tuple[torch.Tensor, int]],):
      ref_wav = load_audio(ref_audio, SAMPLING_RATE)
      ref_rms = float(np.sqrt(np.mean(ref_wav**2)))
      if 0 < ref_rms < 0.1:
          ref_wav = ref_wav * 0.1 / ref_rms

      ref_duration = ref_wav.shape[-1] / SAMPLING_RATE
      if ref_duration > 20.0: # todo just limit it to 20s on front end?
          print(
              "Reference audio is %.1fs long (>20s). This may cause slower "
              "generation, higher memory usage, and degraded voice cloning "
              "quality. We recommend trimming it to 3-10s.",
              ref_duration,
          )

      chunk_size = self.audio_tokenizer.config.hop_length
      clip_size = int(ref_wav.shape[-1] % chunk_size)
      ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav
      # numpy → torch at tokenizer boundary
      ref_wav_tensor = torch.from_numpy(ref_wav).to("mps")
      ref_audio_tokens = encode(self.audio_tokenizer, ref_wav_tensor.unsqueeze(0),).squeeze(0)

      return ref_audio_tokens

  def _decode_and_post_process(
      self,
      tokens: Union[torch.Tensor, List[torch.Tensor]],
  ) -> np.ndarray:
      chunk_audios = [decode(self.audio_tokenizer, t.to("mps").unsqueeze(0))[0].cpu().numpy() for t in tokens]
      audio_waveform = np.concatenate(chunk_audios, axis=-1)
      return audio_waveform.squeeze(0)
      

  def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens):
      ref_weight = sum(CHAR_WEIGHTS[ord(c)] for c in ref_text)
      speed_factor = ref_weight / num_ref_audio_tokens
      target_weight = sum(CHAR_WEIGHTS[ord(c)] for c in text)
      estimated_duration = target_weight / speed_factor
      return int(estimated_duration)

  def _prepare_inference_inputs(
      self,
      text: str,
      num_target_tokens: int,
      ref_text=None,
      ref_audio_tokens: Optional[torch.Tensor] = None,
  ):  
      # todo add lang / instruct?
      style_text = "<|denoise|><|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
      style_tokens = (torch.tensor([tok.encode(style_text)]).repeat(NUM_AUDIO_CODEBOOK, 1).unsqueeze(0)).to(self.device)  # [1, C, N1]

      # Build text tokens
      full_text = ref_text.strip() + " " + text.strip()
      wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
      text_tokens = (torch.tensor([tok.encode(wrapped_text)]).repeat(NUM_AUDIO_CODEBOOK, 1).unsqueeze(0)).to(self.device)  # [1, C, N2]

      # Target: all MASK
      target_audio_tokens = torch.full((1, NUM_AUDIO_CODEBOOK, num_target_tokens), AUDIO_MASK_ID, dtype=torch.long, device=self.device)

      # Conditional input
      parts = [style_tokens, text_tokens]
      parts.append(ref_audio_tokens.unsqueeze(0).to(self.device))
      parts.append(target_audio_tokens)
      cond_input_ids = torch.cat(parts, dim=2)

      cond_total_length = cond_input_ids.shape[2]
      cond_audio_start_idx = cond_total_length - num_target_tokens
      cond_audio_start_idx -= ref_audio_tokens.size(-1)

      cond_audio_mask = torch.zeros(1, cond_total_length, dtype=torch.bool, device=self.device)
      cond_audio_mask[0, cond_audio_start_idx:] = True

      return cond_input_ids, cond_audio_mask


  def _generate_chunked(
      self, target_length, text, ref_text, ref_audio_tokens
  ) -> List[List[torch.Tensor]]:
      avg_tokens_per_char = target_length / len(text)
      text_chunk_len = int(AUDIO_CHUNK_DURATION * FRAME_RATE / avg_tokens_per_char)

      chunks_small = re.findall(r"[^。，！？；：、.,?]+[。，！？；：、.,?]?", text) # eng and cn gaps
      chunks = [""]
      j = 0
      for i in range(len(chunks_small)):
          if chunks_small[i][0] == " ": chunks_small[i] = chunks_small[i][1:]
          if len(chunks[j]) < text_chunk_len + len(chunks_small[i]):
              chunks[j] += chunks_small[i]
          else:
              chunks.append(chunks_small[i])
              j+=1

      chunk_results = []
      for i in range(len(chunks)):
          target_length = self._estimate_target_tokens(chunks[i], ref_text, ref_audio_tokens.size(-1))
          chunk_results.append(self._generate_iterative(text=chunks[i], target_length=target_length, ref_text=ref_text, ref_audio_tokens=ref_audio_tokens))

      return chunk_results


  def _generate_iterative(
      self, text, target_length, ref_text, ref_audio_tokens
  ) -> List[torch.Tensor]:
      cond_input_ids, cond_audio_mask = self._prepare_inference_inputs(text, target_length, ref_text, ref_audio_tokens)

      c_len = cond_input_ids.size(2)
      batch_input_ids = torch.full((2, NUM_AUDIO_CODEBOOK, c_len), AUDIO_MASK_ID, dtype=torch.long, device=self.device,)
      batch_audio_mask = torch.zeros((2, c_len), dtype=torch.bool, device=self.device)
      batch_attention_mask = torch.zeros((2, 1, c_len, c_len), dtype=torch.bool, device=self.device)

      # Cond (0 ~ B-1)
      batch_input_ids[0, :, :c_len] = cond_input_ids
      batch_audio_mask[0, :c_len] = cond_audio_mask
      batch_attention_mask[0, :, :c_len, :c_len] = True

      # Uncond (B ~ 2B-1)
      batch_input_ids[1, :, :target_length] = cond_input_ids[..., -target_length:]
      batch_audio_mask[1, :target_length] = cond_audio_mask[..., -target_length:]
      batch_attention_mask[1, :, :target_length, :target_length] = True

      pad_diag = torch.arange(target_length, c_len, device=self.device)
      batch_attention_mask[1, :, pad_diag, pad_diag] = True

      tokens = torch.full((1, NUM_AUDIO_CODEBOOK, target_length), AUDIO_MASK_ID, dtype=torch.long, device=self.device,)

      timesteps = torch.linspace(0.0, 1.0, NUM_STEPS + 1)
      timesteps = (T_SHIFT * timesteps / (1 + (T_SHIFT - 1) * timesteps)).tolist()

      total_mask = target_length * NUM_AUDIO_CODEBOOK
      rem = total_mask
      sched = []
      for step in range(NUM_STEPS):
          num = (rem if step == NUM_STEPS - 1 else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem,))
          sched.append(int(num))
          rem -= int(num)

      layer_ids = torch.arange(NUM_AUDIO_CODEBOOK, device=self.device).view(1, -1, 1)

      for step in range(NUM_STEPS):
          batch_logits = self(
              input_ids=batch_input_ids,
              audio_mask=batch_audio_mask,
              attention_mask=batch_attention_mask,
          ).to(torch.float32)

          # Extract real target Logits
          # [1, C, T, V]
          c_logits = batch_logits[0: 1, :, c_len - target_length : c_len, :]
          u_logits = batch_logits[1: 2, :, :target_length, :]

          pred_tokens, scores = self._predict_tokens_with_scoring(c_logits, u_logits)

          scores = scores - (layer_ids * LAYER_PENTALTY_FACTOR)

          scores = _gumbel_sample(scores, POSITION_TEMP)

          sample_tokens = tokens[0: 1, :, :target_length]
          scores.masked_fill_(sample_tokens != AUDIO_MASK_ID, -float("inf"))

          _, topk_idx = torch.topk(scores.flatten(), sched[step])
          flat_tokens = sample_tokens.flatten()
          flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
          sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

          # Update individual slices into batched structure
          tokens[0: 1, :, :target_length] = sample_tokens
          batch_input_ids[0: 1, :, c_len - target_length : c_len] = sample_tokens
          batch_input_ids[1: 2, :, :target_length] = sample_tokens

      return tokens[0, :, : target_length]

  def _predict_tokens_with_scoring(self, c_logits, u_logits):
      c_log_probs = F.log_softmax(c_logits, dim=-1)
      u_log_probs = F.log_softmax(u_logits, dim=-1)
      log_probs = torch.log_softmax(
          c_log_probs + GUIDANCE_SCALE * (c_log_probs - u_log_probs),
          dim=-1,
      )

      log_probs[..., AUDIO_MASK_ID] = -float("inf")
      pred_tokens = log_probs.argmax(dim=-1)

      confidence_scores = log_probs.max(dim=-1)[0]

      return pred_tokens, confidence_scores

import soundfile as sf
import pickle

if __name__ == "__main__":
  model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="mps:0", dtype=torch.float16)

  tiny_model = tiny_omni(model)

  torch.manual_seed(0)
  audio = tiny_model.generate(
      text="Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? 谢谢你",
      ref_audio="voice.wav",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  #pickle.dump(audio, open("short.pkl", "wb"))
  exp = pickle.load(open("short.pkl", "rb"))
  sf.write("out.wav", audio, 24000)
  np.testing.assert_allclose(exp, audio, rtol=1e-5)
  '''
  torch.manual_seed(0)
  audio = tiny_model.generate(
      text = "That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black. That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black",
      ref_audio="voice.wav",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  #pickle.dump(audio, open("long.pkl", "wb"))
  exp = pickle.load(open("long.pkl", "rb"))
  sf.write("out_long.wav", audio, 24000)
  np.testing.assert_allclose(exp, audio, rtol=1e-5)
  '''

  torch.manual_seed(0)
  audio = tiny_model.generate(
      text="Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? 谢谢你",
      ref_audio="voice2.wav",
      ref_text="And eh all of the people, I mean we have the greatest military anywhere in the world, and you saw that, in Iran, where, in one week virtually, we knocked out their entire navy, their entire air force",
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  #pickle.dump(audio, open("short2.pkl", "wb"))
  exp = pickle.load(open("short2.pkl", "rb"))
  sf.write("out2.wav", audio, 24000)
  np.testing.assert_allclose(exp, audio, rtol=1e-5)