import logging
import math
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
torch.manual_seed(42)
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
from transformers.modeling_outputs import ModelOutput
from transformers.models.auto import CONFIG_MAPPING, AutoConfig

from omnivoice.utils.audio import (
    cross_fade_chunks,
    fade_and_pad_audio,
    load_audio,
    remove_silence,
)
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAMES
from omnivoice.utils.text import add_punctuation, chunk_text_punctuation

@dataclass
class VoiceClonePrompt:
    ref_audio_tokens: torch.Tensor  # (C, T)
    ref_text: str
    ref_rms: float


FRAME_RATE = 25
AUDIO_CHUNK_DURATION = 15.0
NUM_STEPS = 32
POSITION_TEMP = 5.0
LAYER_PENTALTY_FACTOR = 5.0
GUIDANCE_SCALE = 2.0
T_SHIFT = 0.1
AUDIO_CHUNKED_THRESHOLD = 30.0

@dataclass
class GenerationTask:
    texts: List[str]
    target_lengths: List[int]
    langs: List[Optional[str]]
    instructs: List[Optional[str]]
    ref_texts: List[Optional[str]]
    ref_audio_tokens: List[Optional[torch.Tensor]]
    ref_rms: List[Optional[float]]

    def get_indices(self):
        threshold = int(AUDIO_CHUNKED_THRESHOLD * FRAME_RATE)
        short_idx = [i for i, l in enumerate(self.target_lengths) if l <= threshold]
        long_idx = [i for i, l in enumerate(self.target_lengths) if l > threshold]
        return short_idx, long_idx

class OmniVoiceConfig(PretrainedConfig):
    model_type = "omnivoice"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        audio_mask_id: int = 1024,
        audio_codebook_weights: Optional[list[float]] = None,
        llm_config: Optional[Union[dict, PretrainedConfig]] = None,
        **kwargs,
    ):

        if isinstance(llm_config, dict):
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)

        self.llm_config = llm_config

        super().__init__(**kwargs)
        self.audio_mask_id = audio_mask_id


def _resolve_model_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)

HIDDEN_SIZE = 1024
NUM_AUDIO_CODEBOOK = 8
AUDIO_VOCAB_SIZE = 1025
AUDIO_CODEBOOK_WEIGHTS = [8, 8, 6, 6, 4, 4, 2, 2]

class OmniVoice(PreTrainedModel):
    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class = OmniVoiceConfig

    def __init__(self, config: OmniVoiceConfig):
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

        model.sampling_rate = model.feature_extractor.sampling_rate

        model.duration_estimator = RuleDurationEstimator()

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

    # -------------------------------------------------------------------
    # Inference API
    # -------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        text: Union[str, list[str]],
        ref_text: Union[str, list[str], None] = None,
        ref_audio=None,
        voice_clone_prompt=None,
        instruct: Union[str, list[str], None] = None,
    ) -> list[np.ndarray]:

        self.eval()

        full_task = self._preprocess_all(
            text=text,
            ref_text=ref_text,
            ref_audio=ref_audio,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
        )

        result = self._generate_chunked(full_task)[0]         
        generated_audios = [self._decode_and_post_process(result, full_task.ref_rms[0])]

        return generated_audios

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, tuple[torch.Tensor, int]],
        ref_text: Optional[str] = None,
    ):

        ref_wav = load_audio(ref_audio, self.sampling_rate)
        ref_rms = float(np.sqrt(np.mean(ref_wav**2)))

        if 0 < ref_rms < 0.1:
            ref_wav = ref_wav * 0.1 / ref_rms

        ref_wav = remove_silence(
            ref_wav,
            self.sampling_rate,
            mid_sil=200,
            lead_sil=100,
            trail_sil=200,
        )

        ref_duration = ref_wav.shape[-1] / self.sampling_rate
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
        ref_audio_tokens = self.audio_tokenizer.encode(
            ref_wav_tensor.unsqueeze(0),
        ).audio_codes.squeeze(
            0
        )  # (C, T)

        ref_text = add_punctuation(ref_text)

        return VoiceClonePrompt(
            ref_audio_tokens=ref_audio_tokens,
            ref_text=ref_text,
            ref_rms=ref_rms,
        )

    def _decode_and_post_process(
        self,
        tokens: Union[torch.Tensor, List[torch.Tensor]],
        rms: Union[float, None],
    ) -> np.ndarray:
        tokenizer_device = self.audio_tokenizer.device
        chunk_audios = [
            self.audio_tokenizer.decode(t.to(tokenizer_device).unsqueeze(0))
            .audio_values[0]
            .cpu()
            .numpy()
            for t in tokens
        ]
        audio_waveform = cross_fade_chunks(chunk_audios, self.sampling_rate)
        return audio_waveform.squeeze(0)

    def _preprocess_all(
        self,
        text: Union[str, list[str]],
        ref_text: Union[str, list[str], None] = None,
        ref_audio: Union[
            str,
            list[str],
            tuple[torch.Tensor, int],
            list[tuple[torch.Tensor, int]],
            None,
        ] = None,
        voice_clone_prompt=None,
        instruct: Union[str, list[str], None] = None,
    ):


        voice_clone_prompt =  self.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,)

        num_target_tokens = self._estimate_target_tokens(
            text,
            ref_text,
            voice_clone_prompt.ref_audio_tokens.size(-1),
        )

        return GenerationTask(
            texts=[text],
            target_lengths=[num_target_tokens],
            langs=[None],
            instructs=[None],
            ref_texts=[ref_text],
            ref_audio_tokens=[voice_clone_prompt.ref_audio_tokens],
            ref_rms=[voice_clone_prompt.ref_rms],
        )

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens):
        est = self.duration_estimator.estimate_duration(text, ref_text, num_ref_audio_tokens)
        return max(1, int(est))

    def _prepare_inference_inputs(
        self,
        text: str,
        num_target_tokens: int,
        ref_text: Optional[str] = None,
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
        target_audio_tokens = torch.full(
            (1, NUM_AUDIO_CODEBOOK, num_target_tokens),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

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
        self, task
    ) -> List[List[torch.Tensor]]:

        avg_tokens_per_char = task.target_lengths[0] / len(task.texts[0])
        text_chunk_len = int(
            AUDIO_CHUNK_DURATION
            * FRAME_RATE
            / avg_tokens_per_char
        )
        chunks = chunk_text_punctuation(
            text=task.texts[0],
            chunk_len=text_chunk_len,
            min_chunk_len=3,
        )

        max_num_chunks = len(chunks)
        chunk_results = [[]]

        def _run_batch(text, ref_audio, ref_text):
            target_lengths = [
                self._estimate_target_tokens(
                    text,
                    ref_text,
                    ref_audio.size(-1),
                )
            ]
            sub_task = GenerationTask(
                texts=[text],
                target_lengths=target_lengths,
                langs=[task.langs[0]],
                instructs=[task.instructs[0]],
                ref_texts=[ref_text],
                ref_audio_tokens=[ref_audio],
                ref_rms=[task.ref_rms[0]]
            )
            gen_tokens = self._generate_iterative(sub_task)
            chunk_results[0].append(gen_tokens[0]) # todo

        for i in range(max_num_chunks):
            _run_batch(
                text=chunks[i],
                ref_audio=task.ref_audio_tokens[0],
                ref_text=task.ref_texts[0],
            )

        return chunk_results


    def _generate_iterative(
        self, task: GenerationTask
    ) -> List[torch.Tensor]:
        cond_input_ids, cond_audio_mask = self._prepare_inference_inputs(
                task.texts[0],
                task.target_lengths[0],
                task.ref_texts[0],
                task.ref_audio_tokens[0])

        c_lens = [cond_input_ids.size(2)]
        max_c_len = max(c_lens)
        pad_id = self.config.audio_mask_id  # Or any other tokens

        batch_input_ids = torch.full(
            (2, NUM_AUDIO_CODEBOOK, max_c_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        batch_audio_mask = torch.zeros(
            (2, max_c_len), dtype=torch.bool, device=self.device
        )
        batch_attention_mask = torch.zeros(
            (2, 1, max_c_len, max_c_len), dtype=torch.bool, device=self.device
        )

        c_len, u_len = c_lens[0], task.target_lengths[0]

        # Cond (0 ~ B-1)
        batch_input_ids[0, :, :c_len] = cond_input_ids
        batch_audio_mask[0, :c_len] = cond_audio_mask
        batch_attention_mask[0, :, :c_len, :c_len] = True

        # Uncond (B ~ 2B-1)
        batch_input_ids[1, :, :u_len] = cond_input_ids[..., -u_len:]
        batch_audio_mask[1, :u_len] = cond_audio_mask[..., -u_len:]
        batch_attention_mask[1, :, :u_len, :u_len] = True
        if max_c_len > u_len:
            pad_diag = torch.arange(u_len, max_c_len, device=self.device)
            batch_attention_mask[1, :, pad_diag, pad_diag] = True

        tokens = torch.full(
            (1, NUM_AUDIO_CODEBOOK, max(task.target_lengths)),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=NUM_STEPS,
            t_shift=T_SHIFT,
        ).tolist()
        total_mask = task.target_lengths[0] * NUM_AUDIO_CODEBOOK
        rem = total_mask
        sched = []
        for step in range(NUM_STEPS):
            num = (rem if step == NUM_STEPS - 1 else min(math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])), rem,))
            sched.append(int(num))
            rem -= int(num)
        schedules = [sched]

        layer_ids = torch.arange(NUM_AUDIO_CODEBOOK, device=self.device).view(1, -1, 1)

        for step in range(NUM_STEPS):
            batch_logits = self(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).to(torch.float32)

            k = schedules[0][step]
            if k <= 0:
                continue

            c_len, t_len = c_lens[0], task.target_lengths[0]

            # Extract real target Logits
            # [1, C, T, V]
            c_logits = batch_logits[0: 1, :, c_len - t_len : c_len, :]
            u_logits = batch_logits[1: 2, :, :t_len, :]

            pred_tokens, scores = self._predict_tokens_with_scoring(
                c_logits, u_logits
            )

            scores = scores - (layer_ids * LAYER_PENTALTY_FACTOR)

            scores = _gumbel_sample(scores, POSITION_TEMP)

            sample_tokens = tokens[0: 1, :, :t_len]
            scores.masked_fill_(
                sample_tokens != self.config.audio_mask_id, -float("inf")
            )

            _, topk_idx = torch.topk(scores.flatten(), k)
            flat_tokens = sample_tokens.flatten()
            flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
            sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

            # Update individual slices into batched structure
            tokens[0: 1, :, :t_len] = sample_tokens
            batch_input_ids[0: 1, :, c_len - t_len : c_len] = sample_tokens
            batch_input_ids[1: 2, :, :t_len] = sample_tokens

        return [tokens[0, :, : task.target_lengths[0]]]

    def _predict_tokens_with_scoring(self, c_logits, u_logits):
        c_log_probs = F.log_softmax(c_logits, dim=-1)
        u_log_probs = F.log_softmax(u_logits, dim=-1)
        log_probs = torch.log_softmax(
            c_log_probs + GUIDANCE_SCALE * (c_log_probs - u_log_probs),
            dim=-1,
        )

        log_probs[..., self.config.audio_mask_id] = -float("inf")
        pred_tokens = log_probs.argmax(dim=-1)

        confidence_scores = log_probs.max(dim=-1)[0]

        return pred_tokens, confidence_scores

def _gumbel_sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scaled_logits = logits / temperature
    u = torch.rand_like(scaled_logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-10) + 1e-10)
    return scaled_logits + gumbel_noise


def _get_time_steps(
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_step: int = 10,
    t_shift: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    timesteps = torch.linspace(t_start, t_end, num_step + 1).to(device)
    timesteps = t_shift * timesteps / (1 + (t_shift - 1) * timesteps)
    return timesteps


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