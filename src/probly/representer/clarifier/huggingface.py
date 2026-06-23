"""Question clarification representers backed by Hugging Face generation models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self, override

from probly.representer.sampler.huggingface import HFTextGenerationSampler, load_model

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from os import PathLike

    from transformers import GenerationConfig, PreTrainedTokenizerBase
    from transformers._typing import GenerativePreTrainedModel

    from probly.representer.sampler.huggingface import TextGenerationInput


CLARIFICATION_PROMPT = (
    "Rephrase the following question to clarify its meaning. "
    "Provide a single, clear interpretation that differs from the original phrasing. "
    "Do not answer the question, only rephrase it.\n\n"
    "Question: {question}\n\nRephrased question:"
)


class HFQuestionClarifier(HFTextGenerationSampler):
    """Sample question clarifications with a Hugging Face text generation model."""

    clarification_prompt: str

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
        clarification_prompt: str = CLARIFICATION_PROMPT,
    ) -> None:
        """Initialize the question clarifier.

        Args:
            model: A transformers text generation model.
            tokenizer: The tokenizer associated with ``model``.
            num_samples: Number of clarifications to sample for each question.
            batch_size: Number of samples per question to generate at once. ``None`` means ``num_samples``.
            apply_chat_template: Whether formatted clarification prompts should be wrapped as chat interactions.
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
            clarification_prompt: Prompt template used to turn each question into a clarification request. The
                template must accept a ``question`` format field.
        """
        super().__init__(
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
        self.clarification_prompt = clarification_prompt

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
        clarification_prompt: str = CLARIFICATION_PROMPT,
    ) -> Self:
        """Load a model by name and initialize a question clarifier.

        Args:
            model_name: Hugging Face model name or local model path.
            num_samples: Number of clarifications to sample for each question.
            cache_dir: Optional Hugging Face cache directory.
            force_download: Whether Hugging Face should re-download files even if cached.
            model_kwargs: Additional keyword arguments forwarded to ``AutoModelForCausalLM.from_pretrained``.
            tokenizer_kwargs: Additional keyword arguments forwarded to ``AutoTokenizer.from_pretrained``.
            batch_size: Number of samples per question to generate at once. ``None`` means ``num_samples``.
            apply_chat_template: Whether formatted clarification prompts should be wrapped as chat interactions.
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
            clarification_prompt: Prompt template used to turn each question into a clarification request. The
                template must accept a ``question`` format field.

        Returns:
            A question clarifier backed by the loaded model and tokenizer.
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
            clarification_prompt=clarification_prompt,
        )

    @override
    def _prepare_flat_inputs(self, inputs: Sequence[TextGenerationInput]) -> list[TextGenerationInput]:
        prepared: list[TextGenerationInput] = []
        for question in inputs:
            if not isinstance(question, str):
                msg = "HFQuestionClarifier inputs must be question strings."
                raise TypeError(msg)
            prompt = self.clarification_prompt.format(question=question)
            if self.apply_chat_template:
                prepared.append([{"role": "user", "content": prompt}])
            else:
                prepared.append(prompt)

        return prepared
