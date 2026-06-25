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

class OmniVoice(PreTrainedModel):
    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class = OmniVoiceConfig

    def __init__(self, config):
        super().__init__(config)
        
        self.llm = AutoModel.from_config(self.config.llm_config)

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

class Qwen3RMSNorm:
  def __init__(self): self.variance_epsilon = 1e-6
  
  def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
    hidden_states = to_tiny(hidden_states)
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.cast(dtypes.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = (hidden_states * tiny_Tensor.rsqrt(variance + self.variance_epsilon)).cast(input_dtype)
    return to_torch(self.weight * hidden_states)
  
def repeat_kv(hidden_states, n_rep: int):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return tiny_Tensor.cat(-x2, x1, dim=-1)

class Qwen3Attention:
  def __init__(self):
    self.head_dim = 128
    self.q_norm = Qwen3RMSNorm()
    self.k_norm = Qwen3RMSNorm()
    self.q_proj = tiny_nn.Linear(in_features=1024, out_features=2048, bias=False)
    self.k_proj = tiny_nn.Linear(in_features=1024, out_features=1024, bias=False)
    self.v_proj = tiny_nn.Linear(in_features=1024, out_features=1024, bias=False)
    self.o_proj = tiny_nn.Linear(in_features=1024, out_features=1024, bias=False)
    self.scaling = 0.08838834764831845
    self.num_key_value_groups = 2

  def __call__(self, hidden_states, position_embeddings, attention_mask):
      hidden_states = to_tiny(hidden_states)
      attention_mask = to_tiny(attention_mask)
      position_embeddings = to_tiny(position_embeddings)

      input_shape = hidden_states.shape[:-1]
      hidden_shape = (*input_shape, -1, self.head_dim)

      x = self.q_proj(hidden_states).view(hidden_shape)

      query_states = self.q_norm(x).transpose(1, 2)
      query_states = to_tiny(query_states)

      x = self.k_proj(hidden_states).view(hidden_shape)

      key_states = self.k_norm(x).transpose(1, 2)
      key_states = to_tiny(key_states)

      value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

      cos, sin = position_embeddings
      query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

      key_states = repeat_kv(key_states, self.num_key_value_groups)
      value_states = repeat_kv(value_states, self.num_key_value_groups)

      attn_output = tiny_Tensor.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask
        ).transpose(1, 2).contiguous()
      
      attn_output = attn_output.reshape(*input_shape, -1).contiguous()
      attn_output = self.o_proj(attn_output)
      return to_torch(attn_output)

class Qwen3RotaryEmbedding:
  def __init__(self):
    self.attention_scaling = 1.0
    #https://github.com/huggingface/transformers/blob/f73cc1b1fe0477053638fc929546bac8b3697007/src/transformers/models/qwen3/modeling_qwen3.py#L130-L132
    self.inv_freq = 1.0 / (1000000 ** (tiny_Tensor.arange(0, 128, 2).cast(dtypes.float) / 128))

  def __call__(self, position_ids):
    position_ids = to_tiny(position_ids)
    inv_freq_expanded = self.inv_freq[None, :, None].cast(dtypes.float).expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].cast(dtypes.float)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = tiny_Tensor.cat(freqs, freqs, dim=-1)
    cos = (emb.cos() * self.attention_scaling).cast(dtypes.float16)
    sin = (emb.sin() * self.attention_scaling).cast(dtypes.float16)
    return to_torch(cos), to_torch(sin)

class Qwen3MLP():
  def __init__(self):
    self.down_proj = tiny_nn.Linear(in_features=3072, out_features=1024, bias=False)
    self.gate_proj = tiny_nn.Linear(in_features=1024, out_features=3072, bias=False)
    self.up_proj = tiny_nn.Linear(in_features=1024, out_features=3072, bias=False)
    self.act_fn = tiny_Tensor.silu
  
  def __call__(self, x):
    x = to_tiny(x)
    return to_torch(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class Qwen3DecoderLayer:
  def __init__(self):
    self.input_layernorm = Qwen3RMSNorm()
    self.self_attn = Qwen3Attention()
    self.post_attention_layernorm = Qwen3RMSNorm()
    self.mlp = Qwen3MLP()
  
  def __call__(
      self,
      hidden_states: torch.Tensor,
      attention_mask: torch.Tensor | None = None,
      position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None
  ) -> torch.Tensor:
      residual = hidden_states
      hidden_states = self.input_layernorm(hidden_states)
      # Self Attention
      hidden_states = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings)
      hidden_states = residual + hidden_states

      # Fully Connected
      residual = hidden_states
      hidden_states = self.post_attention_layernorm(hidden_states)
      hidden_states = self.mlp(hidden_states)
      hidden_states = residual + hidden_states
      return hidden_states


class llm:
  def __init__(self):
    self.embed_tokens = tiny_nn.Embedding(151676, 1024)
    self.norm = Qwen3RMSNorm()
    self.rotary_emb = Qwen3RotaryEmbedding()
    self.layers = []
    for i in range(28):
      self.layers.append(Qwen3DecoderLayer())

  def __call__(
    self,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
  ):
      position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
      position_ids = position_ids.unsqueeze(0)

      hidden_states = inputs_embeds
      position_embeddings = self.rotary_emb(position_ids)

      for decoder_layer in self.layers:
        hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask, position_embeddings=position_embeddings,)

      return self.norm(hidden_states)

class HubertModel:
  def __init__(self):
    self.feature_extractor = HubertFeatureEncoder()
    self.feature_projection = HubertFeatureProjection()
    self.encoder = HubertEncoder()

  def __call__(self, input_values: torch.Tensor | None,):
      extract_features = self.feature_extractor(input_values)
      extract_features = extract_features.transpose(1, 2)
      hidden_states = self.feature_projection(extract_features)
      encoder_outputs = self.encoder(hidden_states)
      return encoder_outputs

class HubertPositionalConvEmbedding:
  def __init__(self):
    self.conv = tiny_nn.Conv1d(in_channels=768, out_channels=768, kernel_size=128, stride=1, padding=64, groups=16)
    self.activation = tiny_Tensor.gelu
  
  def __call__(self, hidden_states):
    hidden_states = to_tiny(hidden_states)
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = self.conv(hidden_states)    
    #https://github.com/huggingface/transformers/blob/c5deba28c83d853a1f63a0ab589a4531346fbcb0/src/transformers/models/hubert/modeling_hubert.py#L102
    hidden_states = hidden_states[:, :, : -1]
    hidden_states = self.activation(hidden_states).transpose(1, 2)
    return to_torch(hidden_states)

class HubertFeedForward:
  def __init__(self):
    self.intermediate_dense = tiny_nn.Linear(in_features=768, out_features=3072, bias=True)
    self.intermediate_act_fn = tiny_Tensor.gelu
    self.output_dense = tiny_nn.Linear(in_features=3072, out_features=768, bias=True)

  def __call__(self, hidden_states):
    hidden_states = to_tiny(hidden_states)
    hidden_states = self.intermediate_dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    hidden_states = self.output_dense(hidden_states)
    return to_torch(hidden_states)


class HubertAttention:
  def __init__(self):
    self.head_dim = 64
    self.q_proj = tiny_nn.Linear(in_features=768, out_features=768, bias=True)
    self.k_proj = tiny_nn.Linear(in_features=768, out_features=768, bias=True)
    self.v_proj = tiny_nn.Linear(in_features=768, out_features=768, bias=True)
    self.out_proj = tiny_nn.Linear(in_features=768, out_features=768, bias=True)
    self.scaling = 0.125
    self.is_causal = False
  
  def __call__(
      self,
      hidden_states: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
      hidden_states = to_tiny(hidden_states)
      input_shape = hidden_states.shape[:-1]

      hidden_shape = (*input_shape, -1, self.head_dim)
      query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
      kv_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
      key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
      value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
      
      attn_output = tiny_Tensor.scaled_dot_product_attention(
              query_states,
              key_states,
              value_states,
          ).transpose(1, 2).contiguous()

      attn_output = attn_output.reshape(*input_shape, -1).contiguous()
      attn_output = self.out_proj(attn_output)

      return to_torch(attn_output), None, None

class HubertEncoderLayer:
  def __init__(self):
    self.attention = HubertAttention()
    self.feed_forward = HubertFeedForward()
    self.layer_norm = tiny_nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    self.final_layer_norm = tiny_nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  def __call__(self, hidden_states):
    attn_residual = hidden_states
    hidden_states, _, _ = self.attention(hidden_states)
    hidden_states = attn_residual + hidden_states
    hidden_states = to_tiny(hidden_states)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = to_torch(hidden_states)
    hidden_states = hidden_states + self.feed_forward(hidden_states)
    hidden_states = to_tiny(hidden_states)
    hidden_states = self.final_layer_norm(hidden_states)
    outputs = (hidden_states,)
    return to_torch(outputs)

class HubertEncoder:
  def __init__(self):
    self.pos_conv_embed = HubertPositionalConvEmbedding()
    self.layer_norm = tiny_nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    self.layers = []
    for i in range(12): self.layers.append(HubertEncoderLayer())
  
  def __call__(self, hidden_states: torch.tensor):
    position_embeddings = self.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings.to(hidden_states.device)
    hidden_states = to_tiny(hidden_states)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = to_torch(hidden_states)

    all_hidden_states = ()
    for layer in self.layers:
        all_hidden_states = all_hidden_states + (hidden_states,)
        layer_outputs = layer(hidden_states)
        hidden_states = layer_outputs[0]
    all_hidden_states = all_hidden_states + (hidden_states,)

    return all_hidden_states

class HubertFeatureProjection:
  def __init__(self):
    self.layer_norm = tiny_nn.LayerNorm(512, eps=1e-05, elementwise_affine=True)
    self.projection = tiny_nn.Linear(in_features=512, out_features=768, bias=True)

  def __call__(self, hidden_states):
      hidden_states = to_tiny(hidden_states)
      hidden_states = self.layer_norm(hidden_states)
      hidden_states = self.projection(hidden_states)
      return to_torch(hidden_states)

class HubertGroupNormConvLayer:
  def __init__(self):
    self.conv = tiny_nn.Conv1d(1, 512, kernel_size=10, stride=5, bias=False)
    self.layer_norm = tiny_nn.GroupNorm(512, 512)
  
  def __call__(self, hidden_states):
    hidden_states = to_tiny(hidden_states)
    hidden_states = self.conv(hidden_states)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = tiny_Tensor.gelu(hidden_states)
    return to_torch(hidden_states)

class HubertNoLayerNormConvLayer:
  def __init__(self): self.conv = tiny_nn.Conv1d(512, 512, kernel_size=3, stride=2, bias=False)
  
  def __call__(self, hidden_states):
    hidden_states = to_tiny(hidden_states)
    hidden_states = self.conv(hidden_states)
    hidden_states = tiny_Tensor.gelu(hidden_states)
    return to_torch(hidden_states)

class HubertFeatureEncoder:
  def __init__(self):
    self.conv_layers = [HubertGroupNormConvLayer()]
    for i in range(1, 7):
      self.conv_layers.append(HubertNoLayerNormConvLayer())
  
  def __call__(self, input_values):
    hidden_states = input_values[:, None]
    for conv_layer in self.conv_layers:
      hidden_states = conv_layer(hidden_states)
    return hidden_states

class HiggsAudioV2TokenizerResidualUnit:
  def __init__(self):
    self.conv1 = tiny_nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv2 = tiny_nn.Conv1d(768, 768, kernel_size=1, stride=1, bias=False)

  def __call__(self, hidden_state: torch.Tensor) -> torch.Tensor:
    hidden_state = to_tiny(hidden_state)
    output_tensor = tiny_Tensor.elu(hidden_state)
    output_tensor = self.conv1(output_tensor)
    output_tensor = tiny_Tensor.elu(output_tensor)
    output_tensor = self.conv2(output_tensor)
    hidden_state = hidden_state + output_tensor
    return to_torch(hidden_state)

class HiggsAudioV2TokenizerSemanticEncoderBlock:
  def __init__(self):
    self.res_units = [HiggsAudioV2TokenizerResidualUnit(), HiggsAudioV2TokenizerResidualUnit()]
    self.conv = tiny_nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1)
  
  def __call__(self, hidden_state: torch.Tensor) -> torch.Tensor:
    for unit in self.res_units:
        hidden_state = unit(hidden_state)
    hidden_state = to_tiny(hidden_state)
    hidden_state = self.conv(hidden_state)
    return to_torch(hidden_state)

class SemanticEncoder:
  def __init__(self):
     self.conv = tiny_nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1, bias=False)
     self.conv_blocks = [HiggsAudioV2TokenizerSemanticEncoderBlock(), HiggsAudioV2TokenizerSemanticEncoderBlock()]
   
  def __call__(self, hidden_state: torch.Tensor) -> torch.Tensor:
    hidden_state = to_tiny(hidden_state)
    hidden_state = self.conv(hidden_state)
    hidden_state = to_torch(hidden_state)
    for block in self.conv_blocks:
        hidden_state = block(hidden_state)
    return hidden_state

  def encode(self, hidden_state: torch.Tensor) -> torch.Tensor:
      hidden_state = to_tiny(hidden_state)
      hidden_state = self.conv(hidden_state)
      hidden_state = to_torch(hidden_state)
      for block in self.conv_blocks:
          hidden_state = block(hidden_state)
      return hidden_state

class Snake1d:
  def __call__(self, hidden_states):
    hidden_states = to_tiny(hidden_states)
    shape = hidden_states.shape
    hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
    hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * tiny_Tensor.sin(self.alpha * hidden_states).pow(2)
    hidden_states = hidden_states.reshape(shape)
    return to_torch(hidden_states)

class DacEncoderBlock:
  def __init__(self, blk, in_ch, out_ch, k, s, p):
    self.res_unit1 = blk.res_unit1
    self.res_unit2 = blk.res_unit2
    self.res_unit3 = blk.res_unit3
    self.snake1 = Snake1d()
    self.conv1 = tiny_nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)

  def __call__(self, hidden_state):
    hidden_state = self.res_unit1(hidden_state)
    hidden_state = self.res_unit2(hidden_state)
    hidden_state = self.snake1(self.res_unit3(hidden_state))
    hidden_state = to_tiny(hidden_state)
    hidden_state = self.conv1(hidden_state)
    return to_torch(hidden_state)

class DacEncoder:
  def __init__(self, enc):
    self.conv1 = nn.Conv1d(1, 64, kernel_size=(7,), stride=(1,), padding=(3,))
    self.conv1.weight = enc.conv1.weight
    self.conv1.bias = enc.conv1.bias
    self.conv2 = nn.Conv1d(2048, 256, kernel_size=(3,), stride=(1,), padding=(1,))
    self.conv2.weight = enc.conv2.weight
    self.conv2.bias = enc.conv2.bias
    self.block = [DacEncoderBlock(enc.block[0], 64, 128, 16, 8, 4),
                  DacEncoderBlock(enc.block[1], 128, 256, 10, 5, 3),
                  DacEncoderBlock(enc.block[2], 256, 512, 8, 4, 2),
                  DacEncoderBlock(enc.block[3], 512, 1024, 4, 2, 1),
                  DacEncoderBlock(enc.block[4], 1024, 2048, 6, 3, 2)]
    self.snake1 = Snake1d()
  
  def __call__(self, hidden_state):
    hidden_state = self.conv1(hidden_state)

    for module in self.block:
        hidden_state = module(hidden_state)

    hidden_state = self.snake1(hidden_state)
    hidden_state = self.conv2(hidden_state)

    return hidden_state
  
  def encode(self, hidden_state):
    hidden_state = self.conv1(hidden_state)
    for module in self.block:
        hidden_state = module(hidden_state)
    hidden_state = self.snake1(hidden_state)
    return self.conv2(hidden_state)

class ConvTranspose1d:
  def __init__(self, conv):
    self.weight = conv.weight
    self.bias = conv.bias
    self.stride = conv.stride
    self.padding = conv.padding
    self.groups = conv.groups
    self.dilation = conv.dilation
    self.kernel_size = conv.kernel_size
    self._output_padding = conv._output_padding
    self.output_padding = conv.output_padding
  
  def __call__(self, input):
    return F.conv_transpose1d( # todo
        input,
        self.weight,
        self.bias,
        self.stride,
        self.padding,
        self.output_padding,
        self.groups,
        self.dilation,
    )

class DacResidualUnit:
  def __init__(self, u):
    self.conv1 = tiny_nn.Conv1d(u.conv1.in_channels, u.conv1.out_channels, kernel_size=u.conv1.kernel_size[0], stride=u.conv1.stride[0], padding=u.conv1.padding[0], dilation=u.conv1.dilation[0])
    self.conv2 = tiny_nn.Conv1d(u.conv1.in_channels, u.conv1.out_channels, kernel_size=u.conv2.kernel_size[0], stride=u.conv1.stride[0], padding=u.conv2.padding[0], dilation=u.conv1.dilation[0])
    self.snake1 = Snake1d()
    self.snake2 = Snake1d()

  def __call__(self, hidden_state):
    hidden_state = to_tiny(hidden_state)
    output_tensor = hidden_state
    output_tensor = self.snake1(output_tensor)
    output_tensor = to_tiny(output_tensor)
    output_tensor = self.conv1(output_tensor)
    output_tensor = self.snake2(output_tensor)
    output_tensor = to_tiny(output_tensor)
    output_tensor = self.conv2(output_tensor)
    padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
    if padding > 0:
        print("does this get hit?")
        exit()
        hidden_state = hidden_state[..., padding:-padding]
    output_tensor = to_tiny(output_tensor)
    output_tensor = hidden_state + output_tensor
    output_tensor = to_torch(output_tensor)
    return output_tensor    

class DacDecoderBlock:
  def __init__(self, blk):
    self.snake1 = Snake1d()
    self.conv_t1 = ConvTranspose1d(blk.conv_t1) # todo
    self.res_unit1 = DacResidualUnit(blk.res_unit1)
    self.res_unit2 = DacResidualUnit(blk.res_unit2)
    self.res_unit3 = DacResidualUnit(blk.res_unit3)
   
  def __call__(self, hidden_state):
    hidden_state = self.snake1(hidden_state)
    hidden_state = self.conv_t1(hidden_state)
    hidden_state = self.res_unit1(hidden_state)
    hidden_state = self.res_unit2(hidden_state)
    hidden_state = self.res_unit3(hidden_state)
    return hidden_state

class DacDecoder:
  def __init__(self, dec):
    self.conv1 = nn.Conv1d(256, 1024, kernel_size=(7,), stride=(1,), padding=(3,))
    self.conv1.weight = dec.conv1.weight
    self.conv1.bias = dec.conv1.bias
    self.conv2 = nn.Conv1d(32, 1, kernel_size=(7,), stride=(1,), padding=(3,))
    self.conv2.weight = dec.conv2.weight
    self.conv2.bias = dec.conv2.bias
    self.block = []
    for i in range(5): self.block.append(DacDecoderBlock(dec.block[i]))
    self.snake1 = Snake1d()
  
  def __call__(self, hidden_state):
      hidden_state = self.conv1(hidden_state)

      for layer in self.block:
          hidden_state = layer(hidden_state)

      hidden_state = self.snake1(hidden_state)
      return self.conv2(hidden_state)

class HiggsAudioV2TokenizerResidualVectorQuantization:
   def __init__(self, q):
    self.quantizers = q.quantizers

class audio_tokenizer:
  def __init__(self, tok):
    self.config = tok.config
    self.semantic_model = HubertModel()
    self.encoder_semantic = SemanticEncoder()
    self.acoustic_encoder = DacEncoder(tok.acoustic_encoder)
    self.acoustic_decoder = DacDecoder(tok.acoustic_decoder)
    self.quantizer = HiggsAudioV2TokenizerResidualVectorQuantization(tok.quantizer)
    self.fc = tiny_nn.Linear(1024, 1024)
    self.fc2 = tiny_nn.Linear(1024, 256)

  # https://github.com/huggingface/transformers/blob/1c75d06e73bf25d48a4379b9452ca009da9cf0a1/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py#L41
  def encode(self, input_values: torch.Tensor) -> torch.Tensor:
    e_semantic_input = self._extract_semantic_features(input_values).detach()
    e_semantic = self.encoder_semantic.encode(e_semantic_input.transpose(1, 2))
    e_acoustic = self.acoustic_encoder.encode(input_values)
    embeddings = torch.cat([e_acoustic.to(e_semantic.device), e_semantic], dim=1)
    embeddings = to_tiny(embeddings)
    embeddings = self.fc(embeddings.transpose(1, 2)).transpose(1, 2)
    embeddings = to_torch(embeddings)
    audio_codes = quantizer_encode(self.quantizer, embeddings)
    audio_codes = audio_codes.transpose(0, 1)
    return audio_codes

  def _extract_semantic_features(self, input_values: torch.FloatTensor) -> torch.FloatTensor:
    input_values = input_values[:, 0, :]
    input_values = F.pad(input_values, (160, 160))
    with torch.no_grad():
        hidden_states = self.semantic_model(input_values)

    stacked = torch.stack([h.to(input_values.device) for h in hidden_states], dim=1)
    semantic_features = stacked.mean(dim=1)
    semantic_features = semantic_features[:, :: self.config.semantic_downsample_factor, :]
    return semantic_features

  def decode(self, audio_codes: torch.Tensor,):
      audio_codes = audio_codes.transpose(0, 1)
      quantized_out = torch.tensor(0.0)
      for i, indices in enumerate(audio_codes):
          quantizer = self.quantizer.quantizers[i]
          quantized = F.embedding(indices, quantizer.codebook.embed)
          quantized = quantizer.project_out(quantized)
          quantized = quantized.permute(0, 2, 1)
          quantized_out = quantized_out + quantized
      quantized = quantized_out
      quantized = to_tiny(quantized)
      quantized_acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
      quantized_acoustic = to_torch(quantized_acoustic)
      hidden_state = self.acoustic_decoder.conv1(quantized_acoustic)

      for layer in self.acoustic_decoder.block:
          hidden_state = layer(hidden_state)

      hidden_state = self.acoustic_decoder.snake1(hidden_state)
      return self.acoustic_decoder.conv2(hidden_state)

class omni:
  def __init__(self, model):
    self.audio_tokenizer = audio_tokenizer(model.audio_tokenizer)
    self.device = model.device
    self.llm = llm()
    self.codebook_layer_offsets = (torch.arange(NUM_AUDIO_CODEBOOK) * AUDIO_VOCAB_SIZE).to(self.device)
    self.audio_embeddings = tiny_nn.Embedding(NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, HIDDEN_SIZE)
    self.audio_heads = tiny_nn.Linear(HIDDEN_SIZE, NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, bias=False)

  def _prepare_embed_inputs(
      self, input_ids: torch.Tensor, audio_mask: torch.Tensor
  ) -> torch.Tensor:
      input_ids = to_tiny(input_ids)

      text_embeds = self.llm.embed_tokens(to_tiny(input_ids)[:, 0, :])

      text_embeds = to_torch(text_embeds)
      input_ids = to_torch(input_ids)

      shifted_ids = (input_ids * audio_mask.unsqueeze(1)) + self.codebook_layer_offsets.view(1, -1, 1)
      shifted_ids = to_tiny(shifted_ids)
      audio_embeds = self.audio_embeddings(shifted_ids).sum(axis=1)
      audio_embeds = to_torch(audio_embeds)
      return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

  def __call__(
      self,
      input_ids: torch.LongTensor,
      audio_mask: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      position_ids: Optional[torch.LongTensor] = None,
  ):
      inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)  
      hidden_states = self.llm(
          inputs_embeds=inputs_embeds,
          attention_mask=attention_mask,
          position_ids=position_ids,
      )

      # Shape: [B, S, C * Vocab]
      batch_size, seq_len, _ = hidden_states.shape
      hidden_states = to_tiny(hidden_states)
      logits_flat = self.audio_heads(hidden_states)
      logits_flat = to_torch(logits_flat)
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
      ref_audio_tokens = self.audio_tokenizer.encode(ref_wav_tensor.unsqueeze(0),).squeeze(0)

      return ref_audio_tokens

  def _decode_and_post_process(
      self,
      tokens: Union[torch.Tensor, List[torch.Tensor]],
  ) -> np.ndarray:
      chunk_audios = [self.audio_tokenizer.decode(t.to("mps").unsqueeze(0))[0].cpu().numpy() for t in tokens]
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
from tinygrad.helpers import fetch
from tinygrad.nn.state import safe_load
from tinygrad import Tensor as tiny_Tensor, dtypes, nn as tiny_nn

def to_torch(x):
  if type(x) == tuple: return tuple(to_torch(y) for y in x)
  if type(x) == torch.Tensor: return x
  if x.dtype == dtypes.float16: return torch.Tensor(x.numpy()).to("mps").to(torch.float16)
  if x.dtype == dtypes.int64: return torch.Tensor(x.numpy()).to("mps").to(torch.int64)
  return torch.Tensor(x.numpy()).to("mps")

def to_tiny(x):
  if type(x) == tuple: return tuple(to_tiny(y) for y in x)
  if type(x) == tiny_Tensor: return x
  return tiny_Tensor(x.cpu().detach().numpy())

if __name__ == "__main__":
  model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="mps:0", dtype=torch.float16)
  tiny_Tensor.manual_seed(0)
  model = omni(model)

  weights = safe_load(fetch("https://huggingface.co/k2-fsa/OmniVoice/resolve/main/model.safetensors"))
  #for w in weights.keys(): print(w, type(weights[w]))

  for i in range(len(model.llm.layers)):
    model.llm.layers[i].post_attention_layernorm.weight = tiny_Tensor(weights[f"llm.layers.{i}.post_attention_layernorm.weight"].numpy())
    model.llm.layers[i].input_layernorm.weight = tiny_Tensor(weights[f"llm.layers.{i}.input_layernorm.weight"].numpy())
    model.llm.layers[i].self_attn.q_norm.weight = tiny_Tensor(weights[f"llm.layers.{i}.self_attn.q_norm.weight"].numpy())
    model.llm.layers[i].self_attn.k_norm.weight = tiny_Tensor(weights[f"llm.layers.{i}.self_attn.k_norm.weight"].numpy())
    model.llm.layers[i].mlp.down_proj.weight = tiny_Tensor(weights[f"llm.layers.{i}.mlp.down_proj.weight"].numpy())
    model.llm.layers[i].mlp.gate_proj.weight = tiny_Tensor(weights[f"llm.layers.{i}.mlp.gate_proj.weight"].numpy())
    model.llm.layers[i].mlp.up_proj.weight = tiny_Tensor(weights[f"llm.layers.{i}.mlp.up_proj.weight"].numpy())
    model.llm.layers[i].self_attn.q_proj.weight = tiny_Tensor(weights[f"llm.layers.{i}.self_attn.q_proj.weight"].numpy())
    model.llm.layers[i].self_attn.k_proj.weight = tiny_Tensor(weights[f"llm.layers.{i}.self_attn.k_proj.weight"].numpy())
    model.llm.layers[i].self_attn.v_proj.weight = tiny_Tensor(weights[f"llm.layers.{i}.self_attn.v_proj.weight"].numpy())
    model.llm.layers[i].self_attn.o_proj.weight = tiny_Tensor(weights[f"llm.layers.{i}.self_attn.o_proj.weight"].numpy())
  model.llm.norm.weight = tiny_Tensor(weights[f"llm.norm.weight"].numpy())
  model.llm.embed_tokens.weight = tiny_Tensor(weights["llm.embed_tokens.weight"].numpy()).cast(dtypes.float16)
  model.audio_heads.weight = tiny_Tensor(weights["audio_heads.weight"].numpy())
  model.audio_embeddings.weight = tiny_Tensor(weights["audio_embeddings.weight"].numpy())
  
  weights = safe_load(fetch("https://huggingface.co/k2-fsa/OmniVoice/resolve/main/audio_tokenizer/model.safetensors"))
  for w in weights.keys(): print(w, type(weights[w]))

  model.audio_tokenizer.semantic_model.feature_extractor.conv_layers[0].layer_norm.weight = tiny_Tensor(weights["semantic_model.feature_extractor.conv_layers.0.layer_norm.weight"].numpy())
  model.audio_tokenizer.semantic_model.feature_extractor.conv_layers[0].layer_norm.bias = tiny_Tensor(weights["semantic_model.feature_extractor.conv_layers.0.layer_norm.bias"].numpy())

  for i in range(len(model.audio_tokenizer.semantic_model.feature_extractor.conv_layers)):
    model.audio_tokenizer.semantic_model.feature_extractor.conv_layers[i].conv.weight = tiny_Tensor(weights[f"semantic_model.feature_extractor.conv_layers.{i}.conv.weight"].numpy())
  
  model.audio_tokenizer.semantic_model.encoder.layer_norm.weight = tiny_Tensor(weights[f"semantic_model.encoder.layer_norm.weight"].numpy())
  model.audio_tokenizer.semantic_model.encoder.layer_norm.bias = tiny_Tensor(weights[f"semantic_model.encoder.layer_norm.bias"].numpy())

  for i in range(len(model.audio_tokenizer.semantic_model.encoder.layers)):
    model.audio_tokenizer.semantic_model.encoder.layers[i].final_layer_norm.weight = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.final_layer_norm.weight"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].final_layer_norm.bias = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.final_layer_norm.bias"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].layer_norm.weight = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.layer_norm.weight"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].layer_norm.bias = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.layer_norm.bias"].numpy())

  for i in range(len(model.audio_tokenizer.semantic_model.encoder.layers)):
    model.audio_tokenizer.semantic_model.encoder.layers[i].attention.q_proj.weight = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.attention.q_proj.weight"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].attention.k_proj.weight = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.attention.k_proj.weight"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].attention.v_proj.weight = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.attention.v_proj.weight"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].attention.out_proj.weight = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.attention.out_proj.weight"].numpy())

    model.audio_tokenizer.semantic_model.encoder.layers[i].attention.q_proj.bias = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.attention.q_proj.bias"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].attention.k_proj.bias = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.attention.k_proj.bias"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].attention.v_proj.bias = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.attention.v_proj.bias"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].attention.out_proj.bias = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.attention.out_proj.bias"].numpy())

    model.audio_tokenizer.semantic_model.encoder.layers[i].feed_forward.intermediate_dense.weight = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.feed_forward.intermediate_dense.weight"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].feed_forward.intermediate_dense.bias = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.feed_forward.intermediate_dense.bias"].numpy())

    model.audio_tokenizer.semantic_model.encoder.layers[i].feed_forward.output_dense.weight = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.feed_forward.output_dense.weight"].numpy())
    model.audio_tokenizer.semantic_model.encoder.layers[i].feed_forward.output_dense.bias = tiny_Tensor(weights[f"semantic_model.encoder.layers.{i}.feed_forward.output_dense.bias"].numpy())

  model.audio_tokenizer.acoustic_encoder.snake1.alpha = tiny_Tensor(weights[f"acoustic_encoder.snake1.alpha"].numpy())
  model.audio_tokenizer.acoustic_decoder.snake1.alpha = tiny_Tensor(weights[f"acoustic_decoder.snake1.alpha"].numpy())
  
  for i in range(len(model.audio_tokenizer.acoustic_encoder.block)):
    model.audio_tokenizer.acoustic_encoder.block[i].snake1.alpha = tiny_Tensor(weights[f"acoustic_encoder.block.{i}.snake1.alpha"].numpy())
    model.audio_tokenizer.acoustic_encoder.block[i].conv1.weight = tiny_Tensor(weights[f"acoustic_encoder.block.{i}.conv1.weight"].numpy())
    model.audio_tokenizer.acoustic_encoder.block[i].conv1.bias = tiny_Tensor(weights[f"acoustic_encoder.block.{i}.conv1.bias"].numpy())

  for i in range(len(model.audio_tokenizer.acoustic_decoder.block)):
    model.audio_tokenizer.acoustic_decoder.block[i].snake1.alpha = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.snake1.alpha"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit1.snake1.alpha = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit1.snake1.alpha"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit1.snake2.alpha = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit1.snake2.alpha"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit2.snake1.alpha = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit2.snake1.alpha"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit2.snake2.alpha = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit2.snake2.alpha"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit3.snake1.alpha = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit3.snake1.alpha"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit3.snake2.alpha = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit3.snake2.alpha"].numpy())

    model.audio_tokenizer.acoustic_decoder.block[i].res_unit1.conv1.weight = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit1.conv1.weight"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit1.conv2.weight = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit1.conv2.weight"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit2.conv1.weight = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit2.conv1.weight"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit2.conv2.weight = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit2.conv2.weight"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit3.conv1.weight = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit3.conv1.weight"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit3.conv2.weight = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit3.conv2.weight"].numpy())

    model.audio_tokenizer.acoustic_decoder.block[i].res_unit1.conv1.bias = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit1.conv1.bias"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit1.conv2.bias = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit1.conv2.bias"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit2.conv1.bias = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit2.conv1.bias"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit2.conv2.bias = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit2.conv2.bias"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit3.conv1.bias = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit3.conv1.bias"].numpy())
    model.audio_tokenizer.acoustic_decoder.block[i].res_unit3.conv2.bias = tiny_Tensor(weights[f"acoustic_decoder.block.{i}.res_unit3.conv2.bias"].numpy())

  model.audio_tokenizer.encoder_semantic.conv.weight = tiny_Tensor(weights[f"encoder_semantic.conv.weight"].numpy())
  model.audio_tokenizer.encoder_semantic.conv_blocks[0].conv.weight = tiny_Tensor(weights[f"decoder_semantic.conv_blocks.0.conv.weight"].numpy())
  model.audio_tokenizer.encoder_semantic.conv_blocks[0].conv.bias = tiny_Tensor(weights[f"decoder_semantic.conv_blocks.0.conv.bias"].numpy())
  model.audio_tokenizer.encoder_semantic.conv_blocks[1].conv.weight = tiny_Tensor(weights[f"decoder_semantic.conv_blocks.0.conv.weight"].numpy())
  model.audio_tokenizer.encoder_semantic.conv_blocks[1].conv.bias = tiny_Tensor(weights[f"decoder_semantic.conv_blocks.0.conv.bias"].numpy())

  for i in range(len(model.audio_tokenizer.encoder_semantic.conv_blocks)):
    model.audio_tokenizer.encoder_semantic.conv_blocks[i].res_units[0].conv1.weight = tiny_Tensor(weights[f"encoder_semantic.conv_blocks.{i}.res_units.0.conv1.weight"].numpy())
    model.audio_tokenizer.encoder_semantic.conv_blocks[i].res_units[0].conv2.weight = tiny_Tensor(weights[f"encoder_semantic.conv_blocks.{i}.res_units.0.conv2.weight"].numpy())
    model.audio_tokenizer.encoder_semantic.conv_blocks[i].res_units[1].conv1.weight = tiny_Tensor(weights[f"encoder_semantic.conv_blocks.{i}.res_units.1.conv1.weight"].numpy())
    model.audio_tokenizer.encoder_semantic.conv_blocks[i].res_units[1].conv2.weight = tiny_Tensor(weights[f"encoder_semantic.conv_blocks.{i}.res_units.1.conv2.weight"].numpy())

  model.audio_tokenizer.fc.weight = tiny_Tensor(weights["fc.weight"].numpy())
  model.audio_tokenizer.fc.bias = tiny_Tensor(weights["fc.bias"].numpy())
  model.audio_tokenizer.fc2.weight = tiny_Tensor(weights["fc2.weight"].numpy())
  model.audio_tokenizer.fc2.bias = tiny_Tensor(weights["fc2.bias"].numpy())
  model.audio_tokenizer.semantic_model.encoder.pos_conv_embed.conv.bias = tiny_Tensor(weights["semantic_model.encoder.pos_conv_embed.conv.bias"].numpy())
  model.audio_tokenizer.semantic_model.feature_projection.layer_norm.weight = tiny_Tensor(weights["semantic_model.feature_projection.layer_norm.weight"].numpy())
  model.audio_tokenizer.semantic_model.feature_projection.layer_norm.bias = tiny_Tensor(weights["semantic_model.feature_projection.layer_norm.bias"].numpy())
  model.audio_tokenizer.semantic_model.feature_projection.projection.weight = tiny_Tensor(weights["semantic_model.feature_projection.projection.weight"].numpy())
  model.audio_tokenizer.semantic_model.feature_projection.projection.bias = tiny_Tensor(weights["semantic_model.feature_projection.projection.bias"].numpy())
  #from urllib.request import urlopen
  #from safetensors.torch import load_file
  #state_dict = load_file("model.safetensors")

  #for k in state_dict.keys(): model.k = state_dict[k]
  
  tiny_Tensor.manual_seed(0)
  torch.manual_seed(0)
  audio = model.generate(
      text="Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? 谢谢你",
      ref_audio="voice.wav",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  #pickle.dump(audio, open("short.pkl", "wb"))
  exp = pickle.load(open("short.pkl", "rb"))
  sf.write("out.wav", audio, 24000)
  np.testing.assert_allclose(exp, audio, rtol=1e-5)
  '''
  tiny_Tensor.manual_seed(0)
  torch.manual_seed(0)
  audio = model.generate(
      text = "That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black. That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black",
      ref_audio="voice.wav",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  pickle.dump(audio, open("long.pkl", "wb"))
  exp = pickle.load(open("long.pkl", "rb"))
  sf.write("out_long.wav", audio, 24000)
  np.testing.assert_allclose(exp, audio, rtol=1e-5)
  '''
  tiny_Tensor.manual_seed(0)
  torch.manual_seed(0)
  audio = model.generate(
      text="Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? 谢谢你",
      ref_audio="voice2.wav",
      ref_text="And eh all of the people, I mean we have the greatest military anywhere in the world, and you saw that, in Iran, where, in one week virtually, we knocked out their entire navy, their entire air force",
  ) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  #pickle.dump(audio, open("short2.pkl", "wb"))
  exp = pickle.load(open("short2.pkl", "rb"))
  sf.write("out2.wav", audio, 24000)
  np.testing.assert_allclose(exp, audio, rtol=1e-5)


