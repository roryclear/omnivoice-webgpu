import logging
import math
import os
import re
from typing import List, Optional, Union
import itertools
import pickle

import numpy as np
import torch
import torchaudio
torch.manual_seed(420)
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HiggsAudioV2TokenizerModel,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING, AutoConfig

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

HIDDEN_SIZE = 1024
NUM_AUDIO_CODEBOOK = 8
AUDIO_VOCAB_SIZE = 1025
AUDIO_CODEBOOK_WEIGHTS = [8, 8, 6, 6, 4, 4, 2, 2]
AUDIO_MASK_ID = 1024
SAMPLING_RATE = 24000
# saved from getting all chars with https://github.com/k2-fsa/OmniVoice/blob/9948396864cb713b0c2f92495cf4449bd8717127/omnivoice/utils/duration.py#L204
CHAR_WEIGHTS = pickle.load(open('char_weights.pkl', 'rb'))

from mysoundfile import read
def load_waveform(audio_path: str):
    data, sr = read(audio_path, dtype="float32", always_2d=True)
    return data.T, sr  # (T, C) → (C, T)

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

        self.audio_embeddings = nn.Embedding(
            NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE,
            HIDDEN_SIZE,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(NUM_AUDIO_CODEBOOK) * AUDIO_VOCAB_SIZE,
        )

        self.audio_heads = nn.Linear(
            HIDDEN_SIZE,
            NUM_AUDIO_CODEBOOK * AUDIO_VOCAB_SIZE,
            bias=False,
        )

        self.normalized_audio_codebook_weights = [w / sum(AUDIO_CODEBOOK_WEIGHTS) for w in AUDIO_CODEBOOK_WEIGHTS]

        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        logging.disable(logging.INFO)

        # Resolve to local path first; download only if not already cached
        resolved_path = _resolve_model_path(pretrained_model_name_or_path)
        model = super().from_pretrained(resolved_path, *args, **kwargs)


        model.text_tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        audio_tokenizer_path = os.path.join(resolved_path, "audio_tokenizer")

        # higgs-audio-v2-tokenizer does not support MPS
        # (output channels > 65536)
        tokenizer_device = (
            "cpu" if str(model.device).startswith("mps") else model.device
        )
        model.audio_tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(
            audio_tokenizer_path, device_map=tokenizer_device
        )
        model.feature_extractor = AutoFeatureExtractor.from_pretrained(
            audio_tokenizer_path
        )

        return model

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

    def forward(
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
        self.eval()
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
        ref_wav_tensor = torch.from_numpy(ref_wav).to(self.audio_tokenizer.device)
        ref_audio_tokens = self.audio_tokenizer.encode(ref_wav_tensor.unsqueeze(0),).audio_codes.squeeze(0)  # (C, T)


        return ref_audio_tokens

    def _decode_and_post_process(
        self,
        tokens: Union[torch.Tensor, List[torch.Tensor]],
    ) -> np.ndarray:
        tokenizer_device = self.audio_tokenizer.device
        chunk_audios = [self.audio_tokenizer.decode(t.to(tokenizer_device).unsqueeze(0)).audio_values[0].cpu().numpy() for t in tokens]
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

        style_tokens = (
            self.text_tokenizer(style_text, return_tensors="pt")
            .input_ids.repeat(NUM_AUDIO_CODEBOOK, 1)
            .unsqueeze(0)
        ).to(self.device)  # [1, C, N1]

        # Build text tokens
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_tokens = (
            self.text_tokenizer(wrapped_text, return_tensors="pt").input_ids.repeat(NUM_AUDIO_CODEBOOK, 1).unsqueeze(0)
        ).to(self.device)  # [1, C, N2]

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

def _combine_text(text, ref_text: Optional[str] = None) -> str:
    full_text = ref_text.strip() + " " + text.strip()
    # filter out newline / carriage-return characters
    full_text = re.sub(r"[\r\n]+", "", full_text)

    # replace Chinese parentheses with English ones
    full_text = full_text.replace("\uff08", "(").replace("\uff09", ")")

    # collapse consecutive spaces / tabs into a single space
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # remove spaces around chinese characters
    chinese_range = r"[\u4e00-\u9fff]"
    pattern = rf"(?<={chinese_range})\s+|\s+(?={chinese_range})"
    full_text = re.sub(pattern, "", full_text)

    return full_text