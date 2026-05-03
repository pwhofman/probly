"""Text-generation samplers for Hugging Face transformers models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation import (
    GenerateBeamDecoderOnlyOutput,
    GenerateBeamEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
)

from probly.representation.text_generation import (
    TorchTextGeneration,
    TorchTextGenerationSample,
    TorchTokenGeneration,
)
from probly.representer._representer import Representer

if TYPE_CHECKING:
    from os import PathLike

    from transformers import GenerationConfig
    from transformers._typing import GenerativePreTrainedModel

type ChatInteraction = Sequence[Mapping[str, str]]
type TextGenerationInput = str | ChatInteraction

type HFGenerationOutput = (
    GenerateDecoderOnlyOutput
    | GenerateEncoderDecoderOutput
    | GenerateBeamDecoderOnlyOutput
    | GenerateBeamEncoderDecoderOutput
)


def _as_token_id_set(value: int | Sequence[int] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, torch.Tensor):
        return {int(token_id) for token_id in value.detach().cpu().flatten().tolist()}
    if isinstance(value, Sequence) and not isinstance(value, str):
        token_ids: set[int] = set()
        for token_id in value:
            if token_id is not None:
                token_ids.add(int(token_id))
        return token_ids
    return set()


def _is_encoder_decoder_model(model: GenerativePreTrainedModel) -> bool:
    model_config = getattr(model, "config", None)
    return bool(getattr(model_config, "is_encoder_decoder", False))


def _validate_decoder_only_padding(tokenizer: PreTrainedTokenizerBase) -> None:
    if getattr(tokenizer, "padding_side", None) == "left" and getattr(tokenizer, "pad_token_id", None) is not None:
        return

    msg = (
        "Decoder-only batched generation requires a tokenizer configured for left padding with a pad token. "
        "Set tokenizer.padding_side = 'left' and define tokenizer.pad_token/pad_token_id before constructing "
        "the sampler, or use an encoder-decoder model."
    )
    raise ValueError(msg)


def _configure_decoder_only_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    if getattr(tokenizer, "padding_side", None) != "left" and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    if getattr(tokenizer, "pad_token_id", None) is None and hasattr(tokenizer, "pad_token"):
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token


def load_model(
    model_name: str,
    *,
    cache_dir: str | PathLike[str] | None = None,
    force_download: bool = False,
    model_kwargs: Mapping[str, object] | None = None,
    tokenizer_kwargs: Mapping[str, object] | None = None,
) -> tuple[GenerativePreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Hugging Face causal language model and tokenizer for sampling.

    Args:
        model_name: Hugging Face model name or local model path.
        cache_dir: Optional Hugging Face cache directory.
        force_download: Whether Hugging Face should re-download files even if cached.
        model_kwargs: Additional keyword arguments forwarded to ``AutoModelForCausalLM.from_pretrained``.
        tokenizer_kwargs: Additional keyword arguments forwarded to ``AutoTokenizer.from_pretrained``.

    Returns:
        The loaded model and tokenizer.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        force_download=force_download,
        **dict(tokenizer_kwargs or {}),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        force_download=force_download,
        **dict(model_kwargs or {}),
    )

    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        msg = f"Expected a PreTrainedTokenizerBase tokenizer, got {type(tokenizer)}."
        raise TypeError(msg)
    if not isinstance(model, PreTrainedModel):
        msg = f"Expected a PreTrainedModel model, got {type(model)}."
        raise TypeError(msg)
    model = cast("GenerativePreTrainedModel", model)

    if not _is_encoder_decoder_model(model):
        _configure_decoder_only_tokenizer(tokenizer)

    return model, tokenizer


class HFTextGenerationSampler(Representer[Any, Sequence[TextGenerationInput], torch.Tensor, TorchTextGenerationSample]):
    """Sample decoded text generations from a transformers generation model."""

    model: GenerativePreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    num_samples: int
    batch_size: int | None
    apply_chat_template: bool
    add_generation_prompt: bool
    strip_inputs: bool
    do_sample: bool
    temperature: float | None
    max_new_tokens: int | None
    top_k: int | None
    generation_config: GenerationConfig | None
    with_log_likelihood: bool
    length_normalization: bool

    def __init__(
        self,
        model: GenerativePreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_samples: int,
        batch_size: int | None = None,
        apply_chat_template: bool = True,
        add_generation_prompt: bool = True,
        strip_inputs: bool = True,
        do_sample: bool = True,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        top_k: int | None = None,
        generation_config: GenerationConfig | None = None,
        with_log_likelihood: bool = True,
        length_normalization: bool = True,
    ) -> None:
        """Initialize the text generation sampler.

        Args:
            model: A transformers text generation model.
            tokenizer: The tokenizer associated with ``model``.
            num_samples: Number of completions to sample for each input.
            batch_size: Number of samples per input to generate at once. ``None`` means ``num_samples``.
            apply_chat_template: Whether inputs are chat interactions that need tokenizer chat templating.
            add_generation_prompt: Whether to add a generation prompt when applying the chat template.
                Only has an effect if `apply_chat_template=True`.
            strip_inputs: Whether decoded generations should omit decoder-only prompt tokens.
            do_sample: Whether stochastic sampling should be enabled in ``model.generate``.
            temperature: Optional sampling temperature.
            max_new_tokens: Optional maximum number of generated tokens.
            top_k: Optional top-k sampling cutoff.
            generation_config: Optional generation config for less common generation options.
            with_log_likelihood: Whether to compute generated-sequence log likelihoods.
            length_normalization: Whether log likelihoods should be averaged over scored tokens.
        """
        if num_samples <= 0:
            msg = "num_samples must be positive."
            raise ValueError(msg)
        if batch_size is not None and batch_size <= 0:
            msg = "batch_size must be positive when provided."
            raise ValueError(msg)
        if temperature is not None and temperature <= 0:
            msg = "temperature must be positive when provided."
            raise ValueError(msg)
        if max_new_tokens is not None and max_new_tokens <= 0:
            msg = "max_new_tokens must be positive when provided."
            raise ValueError(msg)
        if top_k is not None and top_k <= 0:
            msg = "top_k must be positive when provided."
            raise ValueError(msg)

        self.model = model
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.strip_inputs = strip_inputs
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_k = top_k
        self.generation_config = generation_config
        self.with_log_likelihood = with_log_likelihood
        self.length_normalization = length_normalization

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        num_samples: int,
        *,
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        model_kwargs: Mapping[str, object] | None = None,
        tokenizer_kwargs: Mapping[str, object] | None = None,
        batch_size: int | None = None,
        apply_chat_template: bool = True,
        add_generation_prompt: bool = True,
        strip_inputs: bool = True,
        do_sample: bool = True,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        top_k: int | None = None,
        generation_config: GenerationConfig | None = None,
        with_log_likelihood: bool = True,
        length_normalization: bool = True,
    ) -> Self:
        """Load a model by name and initialize a text generation sampler.

        Args:
            model_name: Hugging Face model name or local model path.
            num_samples: Number of completions to sample for each input.
            cache_dir: Optional Hugging Face cache directory.
            force_download: Whether Hugging Face should re-download files even if cached.
            model_kwargs: Additional keyword arguments forwarded to ``AutoModelForCausalLM.from_pretrained``.
            tokenizer_kwargs: Additional keyword arguments forwarded to ``AutoTokenizer.from_pretrained``.
            batch_size: Number of samples per input to generate at once. ``None`` means ``num_samples``.
            apply_chat_template: Whether inputs are chat interactions that need tokenizer chat templating.
            add_generation_prompt: Whether to add a generation prompt when applying the chat template.
                Only has an effect if `apply_chat_template=True`.
            strip_inputs: Whether decoded generations should omit decoder-only prompt tokens.
            do_sample: Whether stochastic sampling should be enabled in ``model.generate``.
            temperature: Optional sampling temperature.
            max_new_tokens: Optional maximum number of generated tokens.
            top_k: Optional top-k sampling cutoff.
            generation_config: Optional generation config for less common generation options.
            with_log_likelihood: Whether to compute generated-sequence log likelihoods.
            length_normalization: Whether log likelihoods should be averaged over scored tokens.

        Returns:
            A sampler backed by the loaded model and tokenizer.
        """
        model, tokenizer = load_model(
            model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )
        model.eval()  # generation should always be in inference mode
        return cls(
            model=model,
            tokenizer=tokenizer,
            num_samples=num_samples,
            batch_size=batch_size,
            apply_chat_template=apply_chat_template,
            add_generation_prompt=add_generation_prompt,
            strip_inputs=strip_inputs,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            generation_config=generation_config,
            with_log_likelihood=with_log_likelihood,
            length_normalization=length_normalization,
        )

    @property
    def predictor(self) -> GenerativePreTrainedModel:
        """The underlying model used for generation."""
        return self.model

    def _prepare_prompts(self, inputs: Sequence[TextGenerationInput]) -> list[str]:
        if not self.apply_chat_template:
            return list(inputs)  # ty:ignore[invalid-return-type]

        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            msg = "Tokenizer must implement apply_chat_template when apply_chat_template=True."
            raise TypeError(msg)

        return [
            apply_chat_template(interaction, tokenize=False, add_generation_prompt=self.add_generation_prompt)
            for interaction in inputs
        ]

    def _tokenize(self, prompts: Sequence[str]) -> dict[str, object]:
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        device = torch.device(self.model.device)

        if isinstance(tokenized, Mapping):
            return {
                key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in tokenized.items()
            }

        msg = "Tokenizer must return a mapping of model inputs."
        raise TypeError(msg)

    def _generation_kwargs(self, num_return_samples: int | None = None) -> dict[str, object]:
        values: dict[str, object] = {
            "do_sample": self.do_sample,
            "return_dict_in_generate": True,
            "output_scores": self.with_log_likelihood,
        }
        if self.temperature is not None:
            values["temperature"] = self.temperature
        if self.max_new_tokens is not None:
            values["max_new_tokens"] = self.max_new_tokens
        if self.top_k is not None:
            values["top_k"] = self.top_k

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            values["pad_token_id"] = pad_token_id

        if num_return_samples is not None:
            values["num_return_sequences"] = num_return_samples

        if self.generation_config is None:
            return values

        config = copy(self.generation_config)
        for key, value in values.items():
            setattr(config, key, value)
        return {"generation_config": config}

    def _model_stop_token_ids(self) -> set[int]:
        token_ids: set[int] = set()
        for config in (
            self.generation_config,
            getattr(self.model, "generation_config", None),
            getattr(self.model, "config", None),
        ):
            token_ids.update(_as_token_id_set(getattr(config, "eos_token_id", None)))
            token_ids.update(_as_token_id_set(getattr(config, "pad_token_id", None)))
        return token_ids

    def _generate_chunk(self, prompts: Sequence[str], chunk_size: int) -> TorchTextGeneration:
        is_encoder_decoder = _is_encoder_decoder_model(self.model)
        if not is_encoder_decoder:
            _validate_decoder_only_padding(self.tokenizer)

        use_num_return_sequences = self.do_sample or chunk_size == 1
        chunk_prompts = (
            prompts if use_num_return_sequences else [prompt for prompt in prompts for _ in range(chunk_size)]
        )
        tokenized = self._tokenize(chunk_prompts)
        input_ids = tokenized.get("input_ids")
        if not isinstance(input_ids, torch.Tensor):
            msg = "Tokenizer output must include an input_ids tensor."
            raise TypeError(msg)

        generation_kwargs = self._generation_kwargs(chunk_size if use_num_return_sequences else None)

        with torch.inference_mode():
            outputs = self.model.generate(**tokenized, **generation_kwargs)
            if isinstance(outputs, torch.Tensor):
                msg = "model.generate must return a generation output when return_dict_in_generate=True."
                raise TypeError(msg)

        sequences = outputs.sequences
        if self.with_log_likelihood:
            if outputs.scores is None:
                msg = "model.generate must return scores when output_scores=True."
                raise TypeError(msg)
            transition_scores = self.model.compute_transition_scores(
                sequences,
                outputs.scores,
                beam_indices=getattr(outputs, "beam_indices", None),
                normalize_logits=True,
            )
        else:
            transition_scores = torch.empty((sequences.shape[0], 0), dtype=torch.float32, device=sequences.device)

        if self.strip_inputs and not is_encoder_decoder:
            sequences = sequences[:, input_ids.shape[-1] :]

        token_generation = TorchTokenGeneration(sequences=sequences, transition_scores=transition_scores)
        model_stop_token_ids = self._model_stop_token_ids()
        return token_generation.to_text(
            self.tokenizer,
            length_normalization=self.length_normalization,
            stop_token_ids=model_stop_token_ids if len(model_stop_token_ids) > 0 else None,
        ).reshape(len(prompts), chunk_size)

    def represent(self, inputs: Sequence[TextGenerationInput]) -> TorchTextGenerationSample:
        """Sample completions for each input.

        Args:
            inputs: Prompt strings or chat-template compatible interactions.

        Returns:
            A sample with shape ``(num_inputs, num_samples)`` and sample axis ``1``.
        """
        if len(inputs) == 0:
            msg = "inputs must not be empty."
            raise ValueError(msg)

        prompts = self._prepare_prompts(inputs)
        chunk_limit = self.num_samples if self.batch_size is None else min(self.batch_size, self.num_samples)
        remaining = self.num_samples
        chunks: list[TorchTextGeneration] = []

        while remaining > 0:
            chunk_size = min(chunk_limit, remaining)
            chunks.append(self._generate_chunk(prompts, chunk_size))
            remaining -= chunk_size

        text = np.concatenate([chunk.text for chunk in chunks], axis=1)
        log_likelihood = torch.cat([chunk.log_likelihood for chunk in chunks], dim=1)
        return TorchTextGenerationSample(
            tensor=TorchTextGeneration(text=text, log_likelihood=log_likelihood),
            sample_dim=1,
        )
