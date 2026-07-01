import math
import re
import struct
import pickle

import numpy as np

from tinygrad.helpers import partition, fetch
from tinygrad.nn.state import safe_load, load_state_dict
from tinygrad import Tensor, dtypes, nn, TinyJit, Variable
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
  
MAX_LEN = 2000
FRAME_RATE = 25
AUDIO_CHUNK_DURATION = 15.0
NUM_STEPS = 32
POSITION_TEMP = 5.0
LAYER_PENTALTY_FACTOR = 5.0
GUIDANCE_SCALE = 2.0
T_SHIFT = 0.1
AUDIO_CHUNKED_THRESHOLD = 30.0

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

def resample_numpy(data, orig_sr, target_sr):
    # data is always multi-channel, shape (channels, samples)
    duration = data.shape[1] / orig_sr
    orig_times = np.linspace(0, duration, data.shape[1], endpoint=False)
    new_times = np.linspace(0, duration, int(duration * target_sr), endpoint=False)
    
    resampled_channels = [np.interp(new_times, orig_times, channel) for channel in data]
    return np.array(resampled_channels)

def load_audio(audio_path: str, sampling_rate: int):
    data, sr = load_waveform(audio_path)
    data = np.mean(data, axis=0, keepdims=True)
    data = resample_numpy(data, sr, sampling_rate)
    return data

def _gumbel_sample(logits, temperature: float):
    scaled_logits = logits / temperature
    u = Tensor.rand_like(scaled_logits)
    gumbel_noise = -Tensor.log(-Tensor.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise

_NONVERBAL_PATTERN = re.compile(
    r"\[(laughter|sigh|confirmation-en|question-en|question-ah|question-oh|"
    r"question-ei|question-yi|surprise-ah|surprise-oh|surprise-wa|"
    r"surprise-yo|dissatisfaction-hnn)\]"
)

class Qwen3RMSNorm:
  def __init__(self, sz=1024):
    self.variance_epsilon = 1e-6
    self.weight = Tensor.empty(sz)
  
  def __call__(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.cast(dtypes.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = (hidden_states * Tensor.rsqrt(variance + self.variance_epsilon)).cast(input_dtype)
    return self.weight * hidden_states
  
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
    return Tensor.cat(-x2, x1, dim=-1)

class Qwen3Attention:
  def __init__(self):
    self.head_dim = 128
    self.q_norm = Qwen3RMSNorm(sz=128)
    self.k_norm = Qwen3RMSNorm(sz=128)
    self.q_proj = nn.Linear(in_features=1024, out_features=2048, bias=False)
    self.k_proj = nn.Linear(in_features=1024, out_features=1024, bias=False)
    self.v_proj = nn.Linear(in_features=1024, out_features=1024, bias=False)
    self.o_proj = nn.Linear(in_features=2048, out_features=1024, bias=False)
    self.scaling = 0.08838834764831845
    self.num_key_value_groups = 2

  def __call__(self, hidden_states, position_embeddings, attention_mask):
      position_embeddings = position_embeddings
      input_shape = hidden_states.shape[:-1]
      hidden_shape = (*input_shape, -1, self.head_dim)

      x = self.q_proj(hidden_states).view(hidden_shape)
      query_states = self.q_norm(x).transpose(1, 2)

      x = self.k_proj(hidden_states).view(hidden_shape)
      key_states = self.k_norm(x).transpose(1, 2)

      value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

      cos, sin = position_embeddings
      query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

      key_states = repeat_kv(key_states, self.num_key_value_groups)
      value_states = repeat_kv(value_states, self.num_key_value_groups)

      attn_output = Tensor.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask
        ).transpose(1, 2).contiguous()
      
      attn_output = attn_output.reshape(*input_shape, -1).contiguous()
      attn_output = self.o_proj(attn_output)
      return attn_output

class Qwen3RotaryEmbedding:
  def __init__(self):
    self.attention_scaling = 1.0

  def __call__(self, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].cast(dtypes.float).expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].cast(dtypes.float)
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    emb = Tensor.cat(freqs, freqs, dim=-1)
    cos = (emb.cos() * self.attention_scaling).cast(dtypes.float16)
    sin = (emb.sin() * self.attention_scaling).cast(dtypes.float16)
    return cos, sin

class Qwen3MLP():
  def __init__(self):
    self.down_proj = nn.Linear(in_features=3072, out_features=1024, bias=False)
    self.gate_proj = nn.Linear(in_features=1024, out_features=3072, bias=False)
    self.up_proj = nn.Linear(in_features=1024, out_features=3072, bias=False)
    self.act_fn = Tensor.silu
  
  def __call__(self, x): return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen3DecoderLayer:
  def __init__(self):
    self.input_layernorm = Qwen3RMSNorm()
    self.self_attn = Qwen3Attention()
    self.post_attention_layernorm = Qwen3RMSNorm()
    self.mlp = Qwen3MLP()
  
  def __call__(self, hidden_states, attention_mask=None, position_embeddings=None):
      residual = hidden_states
      hidden_states = self.input_layernorm(hidden_states)
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
    self.embed_tokens = nn.Embedding(151676, 1024)
    self.norm = Qwen3RMSNorm()
    self.rotary_emb = Qwen3RotaryEmbedding()
    self.layers = []
    for i in range(28):
      self.layers.append(Qwen3DecoderLayer())

  def __call__(self, inputs_embeds=None):
      position_ids = Tensor.arange(inputs_embeds.shape[1])
      position_ids = position_ids.unsqueeze(0)

      hidden_states = inputs_embeds
      position_embeddings = self.rotary_emb(position_ids)
      
      for decoder_layer in self.layers:
        hidden_states = decoder_layer(hidden_states, attention_mask=None, position_embeddings=position_embeddings,)

      return self.norm(hidden_states)

class HubertModel:
  def __init__(self):
    self.feature_extractor = HubertFeatureEncoder()
    self.feature_projection = HubertFeatureProjection()
    self.encoder = HubertEncoder()

  def __call__(self, input_values):
      extract_features = self.feature_extractor(input_values)
      extract_features = extract_features.transpose(1, 2)
      hidden_states = self.feature_projection(extract_features)
      encoder_outputs = self.encoder(hidden_states)
      return encoder_outputs

class HubertPositionalConvEmbedding:
  def __init__(self):
    self.conv = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=128, stride=1, padding=64, groups=16)
    self.activation = Tensor.gelu
  
  def __call__(self, hidden_states):
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = self.conv(hidden_states)    
    #https://github.com/huggingface/transformers/blob/c5deba28c83d853a1f63a0ab589a4531346fbcb0/src/transformers/models/hubert/modeling_hubert.py#L102
    hidden_states = hidden_states[:, :, : -1]
    hidden_states = self.activation(hidden_states).transpose(1, 2)
    return hidden_states

class HubertFeedForward:
  def __init__(self):
    self.intermediate_dense = nn.Linear(in_features=768, out_features=3072, bias=True)
    self.intermediate_act_fn = Tensor.gelu
    self.output_dense = nn.Linear(in_features=3072, out_features=768, bias=True)

  def __call__(self, hidden_states):
    hidden_states = self.intermediate_dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    hidden_states = self.output_dense(hidden_states)
    return hidden_states


class HubertAttention:
  def __init__(self):
    self.head_dim = 64
    self.q_proj = nn.Linear(in_features=768, out_features=768, bias=True)
    self.k_proj = nn.Linear(in_features=768, out_features=768, bias=True)
    self.v_proj = nn.Linear(in_features=768, out_features=768, bias=True)
    self.out_proj = nn.Linear(in_features=768, out_features=768, bias=True)
    self.scaling = 0.125
    self.is_causal = False
  
  def __call__(self, hidden_states):
      input_shape = hidden_states.shape[:-1]

      hidden_shape = (*input_shape, -1, self.head_dim)
      query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
      kv_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
      key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
      value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
      
      attn_output = Tensor.scaled_dot_product_attention(
              query_states,
              key_states,
              value_states,
          ).transpose(1, 2).contiguous()

      attn_output = attn_output.reshape(*input_shape, -1).contiguous()
      attn_output = self.out_proj(attn_output)

      return attn_output, None, None

class HubertEncoderLayer:
  def __init__(self):
    self.attention = HubertAttention()
    self.feed_forward = HubertFeedForward()
    self.layer_norm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    self.final_layer_norm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  def __call__(self, hidden_states):
    attn_residual = hidden_states
    hidden_states, _, _ = self.attention(hidden_states)
    hidden_states = attn_residual + hidden_states
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = hidden_states + self.feed_forward(hidden_states)
    hidden_states = self.final_layer_norm(hidden_states)
    outputs = (hidden_states,)
    return outputs

class HubertEncoder:
  def __init__(self):
    self.pos_conv_embed = HubertPositionalConvEmbedding()
    self.layer_norm = nn.LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    self.layers = []
    for i in range(12): self.layers.append(HubertEncoderLayer())
  
  def __call__(self, hidden_states):
    position_embeddings = self.pos_conv_embed(hidden_states)
    hidden_states = hidden_states + position_embeddings
    hidden_states = self.layer_norm(hidden_states)

    all_hidden_states = ()
    for layer in self.layers:
        all_hidden_states = all_hidden_states + (hidden_states,)
        layer_outputs = layer(hidden_states)
        hidden_states = layer_outputs[0]
    all_hidden_states = all_hidden_states + (hidden_states,)

    return all_hidden_states

class HubertFeatureProjection:
  def __init__(self):
    self.layer_norm = nn.LayerNorm(512, eps=1e-05, elementwise_affine=True)
    self.projection = nn.Linear(in_features=512, out_features=768, bias=True)

  def __call__(self, hidden_states):
      hidden_states = self.layer_norm(hidden_states)
      hidden_states = self.projection(hidden_states)
      return hidden_states

class HubertGroupNormConvLayer:
  def __init__(self):
    self.conv = nn.Conv1d(1, 512, kernel_size=10, stride=5, bias=False)
    self.layer_norm = nn.GroupNorm(512, 512)
  
  def __call__(self, hidden_states):
    hidden_states = self.conv(hidden_states)
    hidden_states = self.layer_norm(hidden_states)
    hidden_states = Tensor.gelu(hidden_states)
    return hidden_states

class HubertNoLayerNormConvLayer:
  def __init__(self, k=3): self.conv = nn.Conv1d(512, 512, kernel_size=k, stride=2, bias=False)
  
  def __call__(self, hidden_states):
    hidden_states = self.conv(hidden_states)
    hidden_states = Tensor.gelu(hidden_states)
    return hidden_states

class HubertFeatureEncoder:
  def __init__(self):
    self.conv_layers = [HubertGroupNormConvLayer()]
    for i in range(1, 7):
      self.conv_layers.append(HubertNoLayerNormConvLayer(k=3 if i < 5 else 2))
  
  def __call__(self, input_values):
    hidden_states = input_values[:, None]
    for conv_layer in self.conv_layers:
      hidden_states = conv_layer(hidden_states)
    return hidden_states

class HiggsAudioV2TokenizerResidualUnit:
  def __init__(self):
    self.conv1 = nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv2 = nn.Conv1d(768, 768, kernel_size=1, stride=1, bias=False)

  def __call__(self, hidden_state):
    output_tensor = Tensor.elu(hidden_state)
    output_tensor = self.conv1(output_tensor)
    output_tensor = Tensor.elu(output_tensor)
    output_tensor = self.conv2(output_tensor)
    hidden_state = hidden_state + output_tensor
    return hidden_state

class HiggsAudioV2TokenizerSemanticEncoderBlock:
  def __init__(self):
    self.res_units = [HiggsAudioV2TokenizerResidualUnit(), HiggsAudioV2TokenizerResidualUnit()]
    self.conv = nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1)
  
  def __call__(self, hidden_state):
    for unit in self.res_units:
        hidden_state = unit(hidden_state)
    hidden_state = self.conv(hidden_state)
    return hidden_state

class SemanticEncoder:
  def __init__(self):
     self.conv = nn.Conv1d(768, 768, kernel_size=3, stride=1, padding=1, bias=False)
     self.conv_blocks = [HiggsAudioV2TokenizerSemanticEncoderBlock(), HiggsAudioV2TokenizerSemanticEncoderBlock()]
   
  def __call__(self, hidden_state):
    hidden_state = self.conv(hidden_state)
    for block in self.conv_blocks:
        hidden_state = block(hidden_state)
    return hidden_state

class Snake1d:
  def __init__(self, sz): self.alpha = Tensor.zeros(1, sz, 1)

  def __call__(self, hidden_states):
    shape = hidden_states.shape
    hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
    hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * Tensor.sin(self.alpha * hidden_states).pow(2)
    hidden_states = hidden_states.reshape(shape)
    return hidden_states

class DacEncoderBlock:
  def __init__(self, in_ch, out_ch, k, s, p):
    self.res_unit1 = DacResidualUnit(in_ch=in_ch, out_ch=in_ch, p1=3, d1=1)
    self.res_unit2 = DacResidualUnit(in_ch=in_ch, out_ch=in_ch, p1=9, d1=3)
    self.res_unit3 = DacResidualUnit(in_ch=in_ch, out_ch=in_ch, p1=27, d1=9)
    self.snake1 = Snake1d(in_ch)
    self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)

  def __call__(self, hidden_state):
    hidden_state = self.res_unit1(hidden_state)
    hidden_state = self.res_unit2(hidden_state)
    hidden_state = self.snake1(self.res_unit3(hidden_state))
    hidden_state = self.conv1(hidden_state)
    return hidden_state

class DacEncoder:
  def __init__(self):
    self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=1, padding=3)
    self.conv2 = nn.Conv1d(2048, 256, kernel_size=3, stride=1, padding=1)
    self.block = [DacEncoderBlock(64, 128, 16, 8, 4,),
                  DacEncoderBlock(128, 256, 10, 5, 3),
                  DacEncoderBlock(256, 512, 8, 4, 2),
                  DacEncoderBlock(512, 1024, 4, 2, 1),
                  DacEncoderBlock(1024, 2048, 6, 3, 2)]
    self.snake1 = Snake1d(2048)
  
  def __call__(self, hidden_state):
    hidden_state = self.conv1(hidden_state)
    for module in self.block:
        hidden_state = module(hidden_state)
    hidden_state = self.snake1(hidden_state)
    hidden_state = self.conv2(hidden_state)
    return hidden_state

class ConvTranspose1d:
  def __init__(self, in_ch, n, s, p, op):
    self.weight = Tensor.zeros([in_ch*2, in_ch, n], dtype=dtypes.float32)
    self.bias = Tensor.zeros([in_ch], dtype=dtypes.float32)
    self.stride = s
    self.padding = p
    self.kernel_size = self.padding*2
    self.output_padding = op
  
  def __call__(self, input):
    batch_size, in_channels, in_width = input.shape
    _, _, kernel_size = self.weight.shape

    upsampled = Tensor.zeros(batch_size, in_channels, in_width * self.stride - (self.stride - 1))
    upsampled[:, :, ::self.stride] = input

    pad = 1 * (kernel_size - 1) - self.padding
    weight_flipped = self.weight.flip(-1)
    weight_conv = weight_flipped.permute(1, 0, 2)


    out = Tensor.conv2d(
        upsampled.unsqueeze(2),
        weight_conv.unsqueeze(2),
        bias=None,
        stride=(1, 1),
        padding=(0, pad),
        dilation=(1, 1),
        groups=1,
    ).squeeze(2)

    if self.output_padding > 0: out = Tensor.pad(out, (0, self.output_padding))
    out += self.bias.view(1, -1, 1)
    return out

class DacResidualUnit:
  def __init__(self, in_ch, out_ch, p1, d1):
    self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=1, padding=p1, dilation=d1)
    self.conv2 = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, dilation=1)
    self.snake1 = Snake1d(in_ch)
    self.snake2 = Snake1d(in_ch)

  def __call__(self, hidden_state):
    output_tensor = hidden_state
    output_tensor = self.snake1(output_tensor)
    output_tensor = self.conv1(output_tensor)
    output_tensor = self.snake2(output_tensor)
    output_tensor = self.conv2(output_tensor)
    padding = (hidden_state.shape[-1] - output_tensor.shape[-1]) // 2
    if padding > 0:
        print("does this get hit?")
        exit()
        hidden_state = hidden_state[..., padding:-padding]
    output_tensor = hidden_state + output_tensor
    return output_tensor    

class DacDecoderBlock:
  def __init__(self, in_ch, n, s, p, op):
    self.snake1 = Snake1d(in_ch*2)
    self.conv_t1 = ConvTranspose1d(in_ch=in_ch, n=n, s=s, p=p, op=op) # todo
    self.res_unit1 = DacResidualUnit(out_ch=in_ch, in_ch=in_ch, p1=3, d1=1)
    self.res_unit2 = DacResidualUnit(out_ch=in_ch, in_ch=in_ch, p1=9, d1=3)
    self.res_unit3 = DacResidualUnit(out_ch=in_ch, in_ch=in_ch, p1=27, d1=9)
   
  def __call__(self, hidden_state):
    hidden_state = self.snake1(hidden_state)
    hidden_state = self.conv_t1(hidden_state)
    hidden_state = self.res_unit1(hidden_state)
    hidden_state = self.res_unit2(hidden_state)
    hidden_state = self.res_unit3(hidden_state)
    return hidden_state

class DacDecoder:
  def __init__(self):
    self.conv1 = nn.Conv1d(256, 1024, kernel_size=7, stride=1, padding=3)
    self.conv2 = nn.Conv1d(32, 1, kernel_size=7, stride=1, padding=3)
    self.block = [DacDecoderBlock(512, 16, s=8, p=4, op=0),
                  DacDecoderBlock(256, 10, s=5, p=3, op=1),
                  DacDecoderBlock(128, 8, s=4, p=2, op=0),
                  DacDecoderBlock(64, 4, s=2, p=1, op=0),
                  DacDecoderBlock(32, 6, s=3, p=2, op=1)]
    self.snake1 = Snake1d(32)
  
  def __call__(self, hidden_state):
      hidden_state = self.conv1(hidden_state)
      for layer in self.block:
          hidden_state = layer(hidden_state)
      hidden_state = self.snake1(hidden_state)
      hidden_state = self.conv2(hidden_state)
      return hidden_state

class HiggsAudioV2TokenizerEuclideanCodebook:
  def __init__(self): self.embed = nn.Embedding(1024, 64)

class HiggsAudioV2TokenizerVectorQuantization:
  def __init__(self):
    self.codebook = HiggsAudioV2TokenizerEuclideanCodebook()
    self.project_in = nn.Linear(in_features=1024, out_features=64, bias=True)
    self.project_out = nn.Linear(in_features=64, out_features=1024, bias=True)

class HiggsAudioV2TokenizerResidualVectorQuantization:
  def __init__(self):
    self.quantizers = []
    for _ in range(8): self.quantizers.append(HiggsAudioV2TokenizerVectorQuantization())

  def encode(self, embeddings):
    residual = embeddings
    all_indices = []
    for q in self.quantizers:
      hidden_states = residual.permute(0, 2, 1)
      hidden_states = q.project_in(hidden_states)

      shape = hidden_states.shape
      hidden_states = hidden_states.reshape((-1, shape[-1]))

      embed = q.codebook.embed.weight.T
      scaled_states = hidden_states.pow(2).sum(1, keepdim=True)
      dist = -(scaled_states - 2 * hidden_states @ embed + embed.pow(2).sum(0, keepdim=True))
      indices = dist.argmax(axis=-1)

      indices = indices.view(*shape[:-1])
      quantized = q.codebook.embed(indices)
      quantized = q.project_out(quantized)
      quantized = quantized.permute(0, 2, 1)
      residual = residual - quantized
      all_indices.append(indices)
    out_indices = Tensor.stack(all_indices)
    return out_indices

class audio_tokenizer:
  def __init__(self):
    self.semantic_downsample_factor = 3
    self.hop_length = 960
    self.semantic_model = HubertModel()
    self.encoder_semantic = SemanticEncoder()
    self.acoustic_encoder = DacEncoder()
    self.acoustic_decoder = DacDecoder()
    self.quantizer = HiggsAudioV2TokenizerResidualVectorQuantization()
    self.fc = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, 256)

  # https://github.com/huggingface/transformers/blob/1c75d06e73bf25d48a4379b9452ca009da9cf0a1/src/transformers/models/higgs_audio_v2_tokenizer/modeling_higgs_audio_v2_tokenizer.py#L41
  def encode(self, input_values):
    e_semantic_input = self._extract_semantic_features(input_values).detach()
    e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
    e_acoustic = self.acoustic_encoder(input_values)
    embeddings = Tensor.cat(e_acoustic, e_semantic, dim=1)
    embeddings = self.fc(embeddings.transpose(1, 2)).transpose(1, 2)
    audio_codes = self.quantizer.encode(embeddings)
    audio_codes = audio_codes.transpose(0, 1)
    return audio_codes

  def _extract_semantic_features(self, input_values):
    input_values = input_values[:, 0, :]
    input_values = Tensor.pad(input_values, (160, 160))
    hidden_states = self.semantic_model(input_values)
    stacked = Tensor.stack([h for h in hidden_states], dim=1)
    semantic_features = stacked.mean(axis=1)
    semantic_features = semantic_features[:, :: self.semantic_downsample_factor, :]
    return semantic_features

  def decode(self, audio_codes,):
      audio_codes = audio_codes.transpose(0, 1)
      quantized_out = 0.0
      for i, indices in enumerate(audio_codes):
        quantizer = self.quantizer.quantizers[i]
        quantized = quantizer.codebook.embed(indices)
        quantized = quantizer.project_out(quantized)
        quantized = quantized.permute(0, 2, 1)
        quantized_out = quantized_out + quantized
      quantized = quantized_out
      quantized_acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
      hidden_state = self.acoustic_decoder.conv1(quantized_acoustic)

      for layer in self.acoustic_decoder.block:
          hidden_state = layer(hidden_state)

      hidden_state = self.acoustic_decoder.snake1(hidden_state)
      hidden_state = self.acoustic_decoder.conv2(hidden_state)
      return hidden_state

class omni:
  def __init__(self):
    self.llm = llm()
    self.audio_embeddings = nn.Embedding(NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, HIDDEN_SIZE)
    self.audio_heads = nn.Linear(HIDDEN_SIZE, NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE, bias=False)
    weights = safe_load(fetch("https://huggingface.co/k2-fsa/OmniVoice/resolve/main/model.safetensors"))
    load_state_dict(self, weights)
    #https://github.com/huggingface/transformers/blob/f73cc1b1fe0477053638fc929546bac8b3697007/src/transformers/models/qwen3/modeling_qwen3.py#L130-L132
    self.llm.rotary_emb.inv_freq = 1.0 / (1000000 ** (Tensor.arange(0, 128, 2).cast(dtypes.float) / 128))

    self.audio_tokenizer = audio_tokenizer()
    weights = safe_load(fetch("https://huggingface.co/k2-fsa/OmniVoice/resolve/main/audio_tokenizer/model.safetensors"))
    for k in weights.keys(): print(k)
    weights["semantic_model.encoder.pos_conv_embed.conv.weight"] = weights["semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1"]
    for i in range(len(self.audio_tokenizer.quantizer.quantizers)):
      weights[f"quantizer.quantizers.{i}.codebook.embed.weight"] = weights[f"quantizer.quantizers.{i}.codebook.embed"]
    load_state_dict(self.audio_tokenizer, weights)
    self.codebook_layer_offsets = (Tensor.arange(NUM_AUDIO_CODEBOOK) * AUDIO_VOCAB_SIZE)

  @TinyJit
  def __call__(self, input_ids, audio_mask, c_len, target_length, tokens):
      text_embeds = self.llm.embed_tokens(input_ids[:, 0, :])
      shifted_ids = (input_ids * audio_mask.unsqueeze(1)) + self.codebook_layer_offsets.view(1, -1, 1)
      audio_embeds = self.audio_embeddings(shifted_ids).sum(axis=1)
      inputs_embeds = Tensor.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)
      hidden_states = self.llm(inputs_embeds=inputs_embeds)

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
  
      c_logits = audio_logits[0: 1, :, c_len - target_length : c_len, :]
      u_logits = audio_logits[1: 2, :, :target_length, :]

      pred_tokens, scores = self._predict_tokens_with_scoring(c_logits, u_logits)
      layer_ids = Tensor.arange(NUM_AUDIO_CODEBOOK).view(1, -1, 1)
      scores = scores - (layer_ids * LAYER_PENTALTY_FACTOR)
      scores = _gumbel_sample(scores, POSITION_TEMP)
      scores = Tensor.where(tokens.reshape(NUM_AUDIO_CODEBOOK, target_length) == AUDIO_MASK_ID, scores, -float("inf"))
      return pred_tokens.flatten(), scores.flatten()

  def generate(self, text=None, ref_text=None, ref_audio=None):
    ref_audio_tokens = self.create_voice_clone_prompt(ref_audio=ref_audio)
    num_target_tokens = self._estimate_target_tokens(text, ref_text, ref_audio_tokens.size(-1),)

    result = self._generate_chunked(target_length=num_target_tokens, text=text, ref_text=ref_text, ref_audio_tokens=ref_audio_tokens) 
    return self._decode_and_post_process(result)    

  def create_voice_clone_prompt(self, ref_audio):
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

      chunk_size = self.audio_tokenizer.hop_length
      clip_size = int(ref_wav.shape[-1] % chunk_size)
      ref_wav = ref_wav[:, :-clip_size] if clip_size > 0 else ref_wav
      ref_wav_tensor = Tensor(ref_wav.astype(np.float32))
      ref_audio_tokens = self.audio_tokenizer.encode(ref_wav_tensor.unsqueeze(0),).squeeze(0)

      return ref_audio_tokens

  def _decode_and_post_process(self, tokens):
      chunk_audios = [self.audio_tokenizer.decode(t.unsqueeze(0))[0] for t in tokens]
      audio_waveform = Tensor.cat(*chunk_audios, dim=-1)
      return audio_waveform.squeeze(0)

  def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens):
      ref_weight = sum(CHAR_WEIGHTS[ord(c)] for c in ref_text)
      speed_factor = ref_weight / num_ref_audio_tokens
      target_weight = sum(CHAR_WEIGHTS[ord(c)] for c in text)
      estimated_duration = target_weight / speed_factor
      return int(estimated_duration)

  def _prepare_inference_inputs(self, text: str, num_target_tokens: int, ref_text=None, ref_audio_tokens=None,):  
      # todo add lang / instruct?
      style_text = "<|denoise|><|lang_start|>None<|lang_end|><|instruct_start|>None<|instruct_end|>"
      style_tokens = Tensor([tok.encode(style_text)]).repeat(NUM_AUDIO_CODEBOOK, 1).unsqueeze(0)

      # Build text tokens
      full_text = ref_text.strip() + " " + text.strip()
      wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
      text_tokens = Tensor(tok.encode(wrapped_text)).repeat(NUM_AUDIO_CODEBOOK, 1).unsqueeze(0)

      # Target: all MASK
      target_audio_tokens = Tensor.full((1, NUM_AUDIO_CODEBOOK, num_target_tokens), AUDIO_MASK_ID, dtype=dtypes.long)

      # Conditional input
      parts = [style_tokens, text_tokens]
      parts.append(ref_audio_tokens.unsqueeze(0))
      parts.append(target_audio_tokens)
      cond_input_ids = Tensor.cat(*parts, dim=2)
      cond_total_length = cond_input_ids.shape[2]
      cond_audio_start_idx = cond_total_length - num_target_tokens
      cond_audio_start_idx -= ref_audio_tokens.size(-1)

      cond_audio_mask = Tensor.zeros(1, cond_total_length, dtype=dtypes.bool)
      cond_audio_mask[0, cond_audio_start_idx:] = True
      return cond_input_ids, cond_audio_mask


  def _generate_chunked(
      self, target_length, text, ref_text, ref_audio_tokens):
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
      print("CHUNKS", len(chunks))
      chunk_results = []
      for i in range(len(chunks)):
        ret = self._generate_iterative(text=chunks[i], ref_text=ref_text, ref_audio_tokens=ref_audio_tokens)
        chunk_results.append(ret)
      
      return chunk_results


  def _generate_iterative(self, text, ref_text, ref_audio_tokens):
      target_length = self._estimate_target_tokens(text, ref_text, ref_audio_tokens.size(-1))
      cond_input_ids, cond_audio_mask = self._prepare_inference_inputs(text, target_length, ref_text, ref_audio_tokens)

      c_len = cond_input_ids.size(2)
      batch_input_ids = Tensor.full((2, NUM_AUDIO_CODEBOOK, MAX_LEN), AUDIO_MASK_ID, dtype=dtypes.long)
      batch_audio_mask = Tensor.zeros((2, MAX_LEN), dtype=dtypes.bool)
      batch_attention_mask = Tensor.zeros((2, 1, MAX_LEN, MAX_LEN), dtype=dtypes.bool)

      # Cond (0 ~ B-1)
      batch_input_ids[0:, :, 0:c_len] = cond_input_ids[0]
      batch_audio_mask[0:, 0:c_len] = cond_audio_mask[0]
      batch_attention_mask[0, :, 0:c_len, 0:c_len] = True

      # Uncond (B ~ 2B-1)
      batch_input_ids[1, :, :target_length] = cond_input_ids[..., -target_length:].squeeze(0)
      batch_audio_mask[1, :target_length] = cond_audio_mask[..., -target_length:].squeeze(0)
      batch_attention_mask[1, :, 0:target_length, 0:target_length] = True

      pad_diag = Tensor.arange(target_length, MAX_LEN)
      batch_attention_mask[1, :, pad_diag, pad_diag] = True

      tokens = Tensor.full((NUM_AUDIO_CODEBOOK, target_length), AUDIO_MASK_ID, dtype=dtypes.long)

      timesteps = [i / NUM_STEPS for i in range(NUM_STEPS + 1)]
      timesteps = [(T_SHIFT * t) / (1 + (T_SHIFT - 1) * t) for t in timesteps]

      total_mask = target_length * NUM_AUDIO_CODEBOOK
      rem = total_mask
      sched = []
      for step in range(NUM_STEPS):
          num = (rem if step == NUM_STEPS - 1 else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem,))
          sched.append(int(num))
          rem -= int(num)
      print("SCHED =",sched)
      
      tokens = tokens.flatten()
      for step in range(NUM_STEPS):
        print("STEP",step,"of",NUM_STEPS)
        pred_tokens, scores = self(input_ids=batch_input_ids[:, :, 0:c_len], audio_mask=batch_audio_mask[:, 0:c_len]
                                   ,c_len=c_len, target_length=target_length, tokens=tokens.clone())
        
        _, order = Tensor.sort(-scores)
        idx = order[:sched[step]]
        tokens[idx] = pred_tokens[idx].cast(tokens.dtype)

        batch_input_ids[0: 1, :, c_len - target_length : c_len] = tokens.reshape(NUM_AUDIO_CODEBOOK, target_length)
        batch_input_ids[1:2, :, :target_length] = tokens.reshape(NUM_AUDIO_CODEBOOK, target_length)
        batch_input_ids.realize()
      return tokens.reshape(NUM_AUDIO_CODEBOOK, target_length)

  def _predict_tokens_with_scoring(self, c_logits, u_logits):
      c_log_probs = Tensor.log_softmax(c_logits, axis=-1)
      u_log_probs = Tensor.log_softmax(u_logits, axis=-1)
      log_probs = Tensor.log_softmax(c_log_probs + GUIDANCE_SCALE * (c_log_probs - u_log_probs), axis=-1,)
      log_probs = log_probs.clone() # todo why
      log_probs[..., AUDIO_MASK_ID] = -float("inf")
      pred_tokens = log_probs.argmax(axis=-1)
      confidence_scores = log_probs.max(axis=-1)[0]
      return pred_tokens, confidence_scores

import soundfile as sf
import pickle

if __name__ == "__main__":
  model = omni()

  Tensor.manual_seed(4)
  audio = model.generate(
      text="Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? 谢谢你",
      ref_audio="voice.wav",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
  ).numpy()
  #pickle.dump(audio, open("short.pkl", "wb"))
  exp = pickle.load(open("short.pkl", "rb"))
  sf.write("out.wav", audio, 24000)
  #np.testing.assert_allclose(exp, audio, rtol=1e-5)
  exit()
  Tensor.manual_seed(42)
  audio = model.generate(
      text = "That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black.",# That's it, turn the page on the day, walk away 'Cause there's sense in what I say, I'm forty-fifth generation Roman But I don't know 'em or care when I'm spitting So return to your sitting position and listen, it's fitting That I'm miles ahead and they chase me Show your face on TV then we'll see, you can't do half My crew laughs at your rhubarb-and-custard verses You rain down curses, but I'm waving your hearses driving by Streets riding high with the beats in the sky All stare, eyes glazed, garage burnt down The fire raged for forty days and in forty ways But through the blaze, they see it fade The sea of black",
      ref_audio="voice.wav",
      ref_text="Nothing is ever as it seems anymore and simple declarations bring deeper intrigue, which we are now going to have to spend today unpacking",
  ).numpy() # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  #pickle.dump(audio, open("long.pkl", "wb"))
  exp = pickle.load(open("long.pkl", "rb"))
  sf.write("out_long.wav", audio, 24000)
  #np.testing.assert_allclose(exp, audio, rtol=1e-5)
  #exit()
  
  Tensor.manual_seed(0)
  audio = model.generate(
      text="Testing testing one two three, this is made with Omni-Voice. Can you hear me? or not? 谢谢你",
      ref_audio="voice2.wav",
      ref_text="And eh all of the people, I mean we have the greatest military anywhere in the world, and you saw that, in Iran, where, in one week virtually, we knocked out their entire navy, their entire air force",
  ).numpy() # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.
  #pickle.dump(audio, open("short2.pkl", "wb"))
  exp = pickle.load(open("short2.pkl", "rb"))
  sf.write("out2.wav", audio, 24000)
  np.testing.assert_allclose(exp, audio, rtol=1e-5)
