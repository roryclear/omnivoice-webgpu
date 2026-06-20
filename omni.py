import difflib
import logging
import math
import os
import re
from dataclasses import dataclass, fields
from functools import partial
from typing import Any, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

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
    trim_long_audio,
)
from omnivoice.utils.duration import RuleDurationEstimator
from omnivoice.utils.lang_map import LANG_IDS, LANG_NAMES
from omnivoice.utils.text import add_punctuation, chunk_text_punctuation

logger = logging.getLogger(__name__)

@dataclass
class VoiceClonePrompt:
    ref_audio_tokens: torch.Tensor  # (C, T)
    ref_text: str
    ref_rms: float


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


@dataclass
class GenerationTask:
    batch_size: int
    texts: List[str]
    target_lens: List[int]
    langs: List[Optional[str]]
    instructs: List[Optional[str]]
    ref_texts: List[Optional[str]]
    ref_audio_tokens: List[Optional[torch.Tensor]]
    ref_rms: List[Optional[float]]
    speed: Optional[List[float]] = None

    def get_indices(self, config: OmniVoiceGenerationConfig, frame_rate: int):
        threshold = int(config.audio_chunk_threshold * frame_rate)
        short_idx = [i for i, l in enumerate(self.target_lens) if l <= threshold]
        long_idx = [i for i, l in enumerate(self.target_lens) if l > threshold]
        return short_idx, long_idx

    def slice_task(self, indices: List[int]):
        if not indices:
            return None
        return GenerationTask(
            batch_size=len(indices),
            texts=[self.texts[i] for i in indices],
            target_lens=[self.target_lens[i] for i in indices],
            langs=[self.langs[i] for i in indices],
            instructs=[self.instructs[i] for i in indices],
            ref_texts=[self.ref_texts[i] for i in indices],
            ref_audio_tokens=[self.ref_audio_tokens[i] for i in indices],
            ref_rms=[self.ref_rms[i] for i in indices],
            speed=[self.speed[i] for i in indices] if self.speed else None,
        )


@dataclass
class OmniVoiceModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Config & Model
# ---------------------------------------------------------------------------


class OmniVoiceConfig(PretrainedConfig):
    model_type = "omnivoice"
    sub_configs = {"llm_config": AutoConfig}

    def __init__(
        self,
        audio_vocab_size: int = 1025,
        audio_mask_id: int = 1024,
        num_audio_codebook: int = 8,
        audio_codebook_weights: Optional[list[float]] = None,
        llm_config: Optional[Union[dict, PretrainedConfig]] = None,
        **kwargs,
    ):

        if isinstance(llm_config, dict):
            llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)

        self.llm_config = llm_config

        super().__init__(**kwargs)
        self.audio_vocab_size = audio_vocab_size
        self.audio_mask_id = audio_mask_id
        self.num_audio_codebook = num_audio_codebook
        if audio_codebook_weights is None:
            audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]
        self.audio_codebook_weights = audio_codebook_weights


def _resolve_model_path(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path)


class OmniVoice(PreTrainedModel):
    _supports_flex_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    config_class = OmniVoiceConfig

    def __init__(self, config: OmniVoiceConfig, llm: Optional[PreTrainedModel] = None):
        super().__init__(config)

        self.llm = AutoModel.from_config(self.config.llm_config)

        self.audio_embeddings = nn.Embedding(
            config.num_audio_codebook * config.audio_vocab_size,
            self.config.llm_config.hidden_size,
        )
        self.register_buffer(
            "codebook_layer_offsets",
            torch.arange(config.num_audio_codebook) * config.audio_vocab_size,
        )

        self.audio_heads = nn.Linear(
            self.config.llm_config.hidden_size,
            config.num_audio_codebook * config.audio_vocab_size,
            bias=False,
        )

        self.normalized_audio_codebook_weights = [
            w / sum(config.audio_codebook_weights)
            for w in config.audio_codebook_weights
        ]

        self.post_init()

        # Inference-only attributes (set by from_pretrained when not in train mode)
        self.text_tokenizer = None
        self.audio_tokenizer = None
        self.duration_estimator = None
        self.sampling_rate = None
        self._asr_pipe = None

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

    @torch.inference_mode()
    def transcribe(
        self,
        audio: Union[str, tuple],
    ) -> str:
        """Transcribe audio using the loaded Whisper ASR model.

        Args:
            audio: File path or ``(waveform, sample_rate)`` tuple.
                Waveform can be a numpy array or torch.Tensor of shape
                ``(1, T)`` or ``(T,)``.

        Returns:
            Transcribed text.
        """
        if self._asr_pipe is None:
            raise RuntimeError(
                "ASR model is not loaded. Call model.load_asr_model() first."
            )

        if isinstance(audio, str):
            return self._asr_pipe(audio)["text"].strip()
        else:
            waveform, sr = audio
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.cpu().numpy()
            waveform = np.squeeze(waveform)  # (1, T) or (T,) → (T,)
            audio_input = {
                "array": waveform,
                "sampling_rate": sr,
            }
            return self._asr_pipe(audio_input)["text"].strip()

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def _prepare_embed_inputs(
        self, input_ids: torch.Tensor, audio_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepares embeddings from input_ids of shape (batch_size, layers, seq_length).
        Embedding shape is (batch_size, seq_length, hidden_size).
        """
        text_embeds = self.get_input_embeddings()(input_ids[:, 0, :])

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
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        document_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ):

        inputs_embeds = self._prepare_embed_inputs(input_ids, audio_mask)
        

        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            position_ids=position_ids,
        )
        hidden_states = llm_outputs[0]

        loss = None

        # Shape: [B, S, C * Vocab]
        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)
        # Shape: [B, S, C, Vocab] -> [B, C, S, Vocab]
        audio_logits = logits_flat.view(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)

        if labels is not None:

            # audio_logits.permute(0, 3, 1, 2):
            # [Batch, Layer, Seq, Vocab] -> [Batch, Vocab, Layer, Seq]
            # per_token_loss shape: [Batch, Layer, Seq]，ignore -100
            per_token_loss = torch.nn.functional.cross_entropy(
                audio_logits.permute(0, 3, 1, 2),
                labels,
                reduction="none",
                ignore_index=-100,
            )
            # valid_mask shape: [Batch, Layer, Seq]
            valid_mask = (labels != -100).float()

            # layer_means shape: [num_layers]
            layer_means = (per_token_loss * valid_mask).sum(
                dim=(0, 2)
            ) / valid_mask.sum(dim=(0, 2)).clamp(min=1.0)

            weights = torch.tensor(
                self.normalized_audio_codebook_weights, device=audio_logits.device
            )
            loss = (layer_means * weights).sum()

        return OmniVoiceModelOutput(
            loss=loss,
            logits=audio_logits,
        )

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
        **kwargs,
    ) -> list[np.ndarray]:

        gen_config = OmniVoiceGenerationConfig.from_dict(kwargs)
        self.eval()

        full_task = self._preprocess_all(
            text=text,
            ref_text=ref_text,
            ref_audio=ref_audio,
            voice_clone_prompt=voice_clone_prompt,
            instruct=instruct,
            preprocess_prompt=gen_config.preprocess_prompt,
        )

        short_idx, long_idx = full_task.get_indices(
            gen_config, self.audio_tokenizer.config.frame_rate
        )

        results = [None] * full_task.batch_size

        if short_idx:
            short_task = full_task.slice_task(short_idx)
            short_results = self._generate_iterative(short_task, gen_config)
            for idx, res in zip(short_idx, short_results):
                results[idx] = res

        if long_idx:
            long_task = full_task.slice_task(long_idx)
            long_results = self._generate_chunked(long_task, gen_config)
            for idx, res in zip(long_idx, long_results):
                results[idx] = res

        generated_audios = []
        for i in range(full_task.batch_size):
            assert results[i] is not None, f"Result {i} was not generated"
            generated_audios.append(
                self._decode_and_post_process(
                    results[i], full_task.ref_rms[i], gen_config  # type: ignore[arg-type]
                )
            )

        return generated_audios

    def _generate_chunked(
        self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig
    ) -> List[List[torch.Tensor]]:
        all_chunks = []
        for i in range(task.batch_size):
            avg_tokens_per_char = task.target_lens[i] / len(task.texts[i])
            text_chunk_len = int(
                gen_config.audio_chunk_duration
                * self.audio_tokenizer.config.frame_rate
                / avg_tokens_per_char
            )
            chunks = chunk_text_punctuation(
                text=task.texts[i],
                chunk_len=text_chunk_len,
                min_chunk_len=3,
            )
            logger.debug(f"Item {i} chunked into {len(chunks)} pieces: {chunks}")
            all_chunks.append(chunks)

        has_ref = [t is not None for t in task.ref_audio_tokens]
        assert all(has_ref) or not any(has_ref), (
            "Chunked inference requires all items to either have or not have "
            "ref_audio. Mixed ref/non-ref is not supported."
        )

        max_num_chunks = max(len(c) for c in all_chunks)

        # chunk_results[item_idx] = list of generated token tensors per chunk
        chunk_results = [[] for _ in range(task.batch_size)]

        def _run_batch(indices, texts, ref_audios, ref_texts):
            speed_list = task.speed
            target_lens = [
                self._estimate_target_tokens(
                    texts[j],
                    ref_texts[j],
                    ref_audios[j].size(-1) if ref_audios[j] is not None else None,
                    speed=speed_list[i] if speed_list else 1.0,
                )
                for j, i in enumerate(indices)
            ]
            sub_task = GenerationTask(
                batch_size=len(indices),
                texts=texts,
                target_lens=target_lens,
                langs=[task.langs[i] for i in indices],
                instructs=[task.instructs[i] for i in indices],
                ref_texts=ref_texts,
                ref_audio_tokens=ref_audios,
                ref_rms=[task.ref_rms[i] for i in indices],
                speed=[task.speed[i] for i in indices] if task.speed else None,
            )
            gen_tokens = self._generate_iterative(sub_task, gen_config)
            for j, idx in enumerate(indices):
                chunk_results[idx].append(gen_tokens[j])

        if all(has_ref):
            # All items have reference audio.
            # We still sequentially generate chunks within each item, but we
            # batch across items for the same chunk index. This allows to keep
            # the VRAM usage manageable while still benefiting from batching.
            for ci in range(max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[task.ref_audio_tokens[i] for i in indices],
                    ref_texts=[task.ref_texts[i] for i in indices],
                )
        else:
            # No reference audio — generate chunk 0 for all items first,
            # then use chunk 0 output as reference for all subsequent chunks.
            indices_0 = [i for i in range(task.batch_size) if len(all_chunks[i]) > 0]
            _run_batch(
                indices_0,
                texts=[all_chunks[i][0] for i in indices_0],
                ref_audios=[None] * len(indices_0),
                ref_texts=[None] * len(indices_0),
            )
            first_chunk_map = {idx: chunk_results[idx][0] for idx in indices_0}

            # Batch all remaining chunks, using chunk 0 as fixed reference
            for ci in range(1, max_num_chunks):
                indices = [i for i in range(task.batch_size) if ci < len(all_chunks[i])]
                if not indices:
                    continue
                _run_batch(
                    indices,
                    texts=[all_chunks[i][ci] for i in indices],
                    ref_audios=[first_chunk_map[i] for i in indices],
                    ref_texts=[all_chunks[i][0] for i in indices],
                )

        return chunk_results

    def create_voice_clone_prompt(
        self,
        ref_audio: Union[str, tuple[torch.Tensor, int]],
        ref_text: Optional[str] = None,
        preprocess_prompt: bool = True,
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
        gen_config: OmniVoiceGenerationConfig,
    ) -> np.ndarray:
        tokenizer_device = self.audio_tokenizer.device
        if isinstance(tokens, list):
            chunk_audios = [
                self.audio_tokenizer.decode(t.to(tokenizer_device).unsqueeze(0))
                .audio_values[0]
                .cpu()
                .numpy()
                for t in tokens
            ]
            audio_waveform = cross_fade_chunks(chunk_audios, self.sampling_rate)
        else:
            audio_waveform = (
                self.audio_tokenizer.decode(tokens.to(tokenizer_device).unsqueeze(0))
                .audio_values[0]
                .cpu()
                .numpy()
            )
            audio_waveform = self._post_process_audio(
                audio_waveform,
                postprocess_output=gen_config.postprocess_output,
                ref_rms=rms,
            )
        return audio_waveform.squeeze(0)

    def _post_process_audio(
        self,
        generated_audio: np.ndarray,
        postprocess_output: bool,
        ref_rms: Union[float, None],
    ) -> np.ndarray:
        if postprocess_output:
            generated_audio = remove_silence(
                generated_audio,
                self.sampling_rate,
                mid_sil=500,
                lead_sil=100,
                trail_sil=100,
            )

        if ref_rms is not None and ref_rms < 0.1:
            generated_audio = generated_audio * ref_rms / 0.1
        elif ref_rms is None:
            peak = np.abs(generated_audio).max()
            if peak > 1e-6:
                generated_audio = generated_audio / peak * 0.5

        generated_audio = fade_and_pad_audio(
            generated_audio,
            sample_rate=self.sampling_rate,
        )
        return generated_audio

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
        preprocess_prompt: bool = True,
    ) -> GenerationTask:

        text_list = [text]
        batch_size = len(text_list)

        instruct_list = self._ensure_list(instruct, batch_size)

        ref_text_list = self._ensure_list(ref_text, batch_size, auto_repeat=False)
        ref_audio_list = self._ensure_list(ref_audio, batch_size, auto_repeat=False)

        voice_clone_prompt = []
        for i in range(len(ref_text_list)):
            voice_clone_prompt.append(
                self.create_voice_clone_prompt(
                    ref_audio=ref_audio_list[i],
                    ref_text=ref_text_list[i],
                    preprocess_prompt=preprocess_prompt,
                )
            )


        voice_clone_prompt_list = self._ensure_list(voice_clone_prompt, batch_size)

        ref_text_list = [vc.ref_text for vc in voice_clone_prompt_list]
        ref_audio_tokens_list = [
            vc.ref_audio_tokens for vc in voice_clone_prompt_list
        ]
        ref_rms_list = [vc.ref_rms for vc in voice_clone_prompt_list]

        num_target_tokens_list = []
        for i in range(batch_size):
            est = self._estimate_target_tokens(
                text_list[i],
                ref_text_list[i],
                ref_audio_tokens_list[i].size(-1)
                if ref_audio_tokens_list[i] is not None
                else None,
                speed=1.0,
            )
            num_target_tokens_list.append(est)

        return GenerationTask(
            batch_size=batch_size,
            texts=text_list,
            target_lens=num_target_tokens_list,
            langs=[None],
            instructs=instruct_list,
            ref_texts=ref_text_list,
            ref_audio_tokens=ref_audio_tokens_list,
            ref_rms=ref_rms_list,
            speed=None,
        )

    def _estimate_target_tokens(self, text, ref_text, num_ref_audio_tokens, speed=1.0):
        est = self.duration_estimator.estimate_duration(
            text, ref_text, num_ref_audio_tokens
        )
        return max(1, int(est))

    def _ensure_list(
        self, x: Union[Any, List[Any]], batch_size: int, auto_repeat: bool = True
    ) -> List[Any]:
        x_list = x if isinstance(x, list) else [x]
        if len(x_list) not in (
            1,
            batch_size,
        ):
            raise ValueError(
                f"should be either the number of the text or 1, but got {len(x_list)}"
            )
        if auto_repeat and len(x_list) == 1 and batch_size is not None:
            x_list = x_list * batch_size
        return x_list

    def _prepare_inference_inputs(
        self,
        text: str,
        num_target_tokens: int,
        ref_text: Optional[str] = None,
        ref_audio_tokens: Optional[torch.Tensor] = None,
        lang: Optional[str] = None,
        instruct: Optional[str] = None,
        denoise: bool = True,
    ):
        """Prepare input_ids and audio masks for inference.
        Args:
            text: Target text to generate.
            num_target_tokens: Number of audio tokens to generate.
            ref_text: Optional reference text for voice cloning.
            ref_audio_tokens: Optional reference audio tokens for voice cloning.
                with shape (C, T).
            lang: Optional language ID.
            instruct: Optional style instruction for voice design.
            denoise: Whether to include the <|denoise|> token.
        """

        # Build style tokens: <|denoise|> + <|lang_start|>...<|lang_end|>
        #                      + <|instruct_start|>...<|instruct_end|>
        style_text = ""
        if denoise and ref_audio_tokens is not None:
            style_text += "<|denoise|>"
        lang_str = lang if lang else "None"
        instruct_str = instruct if instruct else "None"
        style_text += f"<|lang_start|>{lang_str}<|lang_end|>"
        style_text += f"<|instruct_start|>{instruct_str}<|instruct_end|>"

        style_tokens = (
            self.text_tokenizer(style_text, return_tensors="pt")
            .input_ids.repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(
            self.device
        )  # [1, C, N1]

        # Build text tokens
        full_text = _combine_text(ref_text=ref_text, text=text)
        wrapped_text = f"<|text_start|>{full_text}<|text_end|>"
        text_tokens = (
            _tokenize_with_nonverbal_tags(wrapped_text, self.text_tokenizer)
            .repeat(self.config.num_audio_codebook, 1)
            .unsqueeze(0)
        ).to(
            self.device
        )  # [1, C, N2]

        # Target: all MASK
        target_audio_tokens = torch.full(
            (1, self.config.num_audio_codebook, num_target_tokens),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        # Conditional input
        parts = [style_tokens, text_tokens]
        if ref_audio_tokens is not None:
            parts.append(ref_audio_tokens.unsqueeze(0).to(self.device))
        parts.append(target_audio_tokens)
        cond_input_ids = torch.cat(parts, dim=2)

        cond_total_length = cond_input_ids.shape[2]
        cond_audio_start_idx = cond_total_length - num_target_tokens
        if ref_audio_tokens is not None:
            cond_audio_start_idx -= ref_audio_tokens.size(-1)

        cond_audio_mask = torch.zeros(
            1, cond_total_length, dtype=torch.bool, device=self.device
        )
        cond_audio_mask[0, cond_audio_start_idx:] = True

        return {
            "input_ids": cond_input_ids,
            "audio_mask": cond_audio_mask,
        }

    def _generate_iterative(
        self, task: GenerationTask, gen_config: OmniVoiceGenerationConfig
    ) -> List[torch.Tensor]:
        inputs_list = [
            self._prepare_inference_inputs(
                task.texts[0],
                task.target_lens[0],
                task.ref_texts[0],
                task.ref_audio_tokens[0],
                task.langs[0],
                task.instructs[0],
                gen_config.denoise,
            )
        ]

        c_lens = [inp["input_ids"].size(2) for inp in inputs_list]
        max_c_len = max(c_lens)
        pad_id = self.config.audio_mask_id  # Or any other tokens

        batch_input_ids = torch.full(
            (2, self.config.num_audio_codebook, max_c_len),
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

        for i, inp in enumerate(inputs_list):
            c_len, u_len = c_lens[i], task.target_lens[i]

            # Cond (0 ~ B-1)
            batch_input_ids[i, :, :c_len] = inp["input_ids"]
            batch_audio_mask[i, :c_len] = inp["audio_mask"]
            batch_attention_mask[i, :, :c_len, :c_len] = True

            # Uncond (B ~ 2B-1)
            batch_input_ids[1 + i, :, :u_len] = inp["input_ids"][..., -u_len:]
            batch_audio_mask[1 + i, :u_len] = inp["audio_mask"][..., -u_len:]
            batch_attention_mask[1 + i, :, :u_len, :u_len] = True
            if max_c_len > u_len:
                pad_diag = torch.arange(u_len, max_c_len, device=self.device)
                batch_attention_mask[1 + i, :, pad_diag, pad_diag] = True

        tokens = torch.full(
            (1, self.config.num_audio_codebook, max(task.target_lens)),
            self.config.audio_mask_id,
            dtype=torch.long,
            device=self.device,
        )

        timesteps = _get_time_steps(
            t_start=0.0,
            t_end=1.0,
            num_step=gen_config.num_step,
            t_shift=gen_config.t_shift,
        ).tolist()
        schedules = []
        for t_len in task.target_lens:
            total_mask = t_len * self.config.num_audio_codebook
            rem = total_mask
            sched = []
            for step in range(gen_config.num_step):
                num = (
                    rem
                    if step == gen_config.num_step - 1
                    else min(
                        math.ceil(total_mask * (timesteps[step + 1] - timesteps[step])),
                        rem,
                    )
                )
                sched.append(int(num))
                rem -= int(num)
            schedules.append(sched)

        layer_ids = torch.arange(
            self.config.num_audio_codebook, device=self.device
        ).view(1, -1, 1)

        for step in range(gen_config.num_step):
            batch_logits = self(
                input_ids=batch_input_ids,
                audio_mask=batch_audio_mask,
                attention_mask=batch_attention_mask,
            ).logits.to(torch.float32)

            for i in range(1): # todo remove loop
                k = schedules[i][step]
                if k <= 0:
                    continue

                c_len, t_len = c_lens[i], task.target_lens[i]

                # Extract real target Logits
                # [1, C, T, V]
                c_logits = batch_logits[i : i + 1, :, c_len - t_len : c_len, :]
                u_logits = batch_logits[1 + i : 1 + i + 1, :, :t_len, :]

                pred_tokens, scores = self._predict_tokens_with_scoring(
                    c_logits, u_logits, gen_config
                )

                scores = scores - (layer_ids * gen_config.layer_penalty_factor)

                if gen_config.position_temperature > 0.0:
                    scores = _gumbel_sample(scores, gen_config.position_temperature)

                sample_tokens = tokens[i : i + 1, :, :t_len]
                scores.masked_fill_(
                    sample_tokens != self.config.audio_mask_id, -float("inf")
                )

                _, topk_idx = torch.topk(scores.flatten(), k)
                flat_tokens = sample_tokens.flatten()
                flat_tokens[topk_idx] = pred_tokens.flatten()[topk_idx]
                sample_tokens.copy_(flat_tokens.view_as(sample_tokens))

                # Update individual slices into batched structure
                tokens[i : i + 1, :, :t_len] = sample_tokens
                batch_input_ids[i : i + 1, :, c_len - t_len : c_len] = sample_tokens
                batch_input_ids[1 + i : 1 + i + 1, :, :t_len] = sample_tokens

        return [tokens[i, :, : task.target_lens[i]]]

    def _predict_tokens_with_scoring(self, c_logits, u_logits, gen_config):
        if gen_config.guidance_scale != 0:
            c_log_probs = F.log_softmax(c_logits, dim=-1)
            u_log_probs = F.log_softmax(u_logits, dim=-1)
            log_probs = torch.log_softmax(
                c_log_probs + gen_config.guidance_scale * (c_log_probs - u_log_probs),
                dim=-1,
            )
        else:
            log_probs = F.log_softmax(c_logits, dim=-1)

        log_probs[..., self.config.audio_mask_id] = -float("inf")

        if gen_config.class_temperature > 0.0:
            filtered_probs = _filter_top_k(log_probs, ratio=0.1)
            pred_tokens = _gumbel_sample(
                filtered_probs, gen_config.class_temperature
            ).argmax(dim=-1)
        else:
            pred_tokens = log_probs.argmax(dim=-1)

        confidence_scores = log_probs.max(dim=-1)[0]

        return pred_tokens, confidence_scores


def _filter_top_k(logits: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    k = math.ceil(ratio * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(-1, ind, val)
    return probs


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


def _tokenize_with_nonverbal_tags(text: str, tokenizer) -> torch.Tensor:
    """Tokenize text containing non-verbal tags, handling each tag independently.

    Non-verbal tags are tokenized standalone to guarantee consistent token
    IDs regardless of surrounding language context (Chinese, English, etc.).

    Args:
        text: Full text string potentially containing non-verbal tags.
        tokenizer: HuggingFace text tokenizer instance.
    Returns:
        Token IDs tensor of shape (1, seq_len).
    """
    parts = []
    last_end = 0
    for m in _NONVERBAL_PATTERN.finditer(text):
        if m.start() > last_end:
            segment = text[last_end : m.start()]
            ids = tokenizer(segment, add_special_tokens=False).input_ids
            if ids:
                parts.append(ids)
        tag_ids = tokenizer(m.group(), add_special_tokens=False).input_ids
        if tag_ids:
            parts.append(tag_ids)
        last_end = m.end()
    if last_end < len(text):
        segment = text[last_end:]
        ids = tokenizer(segment, add_special_tokens=False).input_ids
        if ids:
            parts.append(ids)

    if not parts:
        result = tokenizer(text, return_tensors="pt").input_ids
    else:
        combined = []
        for p in parts:
            combined.extend(p)
        result = torch.tensor([combined], dtype=torch.long)
    return result


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