# adapted from https://github.com/idanshen/Self-Distillation

from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments


@dataclass
class DistilConfig(TrainingArguments):
    r"""
    Configuration class for the [`DistilTrainer`].

    This class includes only the parameters that are specific to Distil training. For a full list of training arguments,
    please refer to the [`~transformers.TrainingArguments`] documentation. Note that default values in this class may
    differ from those in [`~transformers.TrainingArguments`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        > Parameters that control the model and reference model

        model_init_kwargs (`str`, `dict[str, Any]`, *optional*):
            Keyword arguments for [`~transformers.AutoModelForCausalLM.from_pretrained`], used when the `model`
            argument of the [`DistilTrainer`] is provided as a string.
        disable_dropout (`bool`, *optional*, defaults to `False`):
            Whether to disable dropout in the model. This is useful for training with a reference model, as it prevents
            the model from generating different logprobs for the same input.

        > Parameters that control the data preprocessing

        remove_unused_columns (`bool`, *optional*, defaults to `False`):
            Whether to only keep the column `"prompt"` in the dataset. If you use a custom reward function that
            requires any column other than `"prompts"` and `"completions"`, you should keep this to `False`.
        max_prompt_length (`int` or `None`, *optional*, defaults to `512`):
            Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.
        num_generations (`int` or `None`, *optional*, defaults to `8`):
            Number of generations per prompt to sample. The effective batch size (num_processes * per_device_batch_size
            * gradient_accumulation_steps) must be evenly divisible by this value.
        max_completion_length (`int` or `None`, *optional*, defaults to `256`):
            Maximum length of the generated completion.
        ds3_gather_for_generation (`bool`, *optional*, defaults to `True`):
            This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
            improving generation speed. However, disabling this option allows training models that exceed the VRAM
            capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
            with vLLM generation.
        shuffle_dataset (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training dataset.

        > Parameters that control generation

        generation_batch_size: (`int`, *optional*):
            Batch size to use for generation. If `None`, it defaults to the effective training batch size:
            `per_device_train_batch_size * num_processes * steps_per_generation`. In other words, there is one
            generation batch processed per optimization step. Mutually exclusive with `steps_per_generation`.
        steps_per_generation: (`int`, *optional*):
            Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`. Mutually exclusive
            with `generation_batch_size`.
        temperature (`float`, defaults to `1.0`):
            Temperature for sampling. The higher the temperature, the more random the completions.
        top_p (`float`, *optional*, defaults to `1.0`):
            Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*):
            Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, top-k-filtering is
            disabled and all tokens are considered.
        min_p (`float`, *optional*):
            Minimum token probability, which will be scaled by the probability of the most likely token. It must be a
            value between `0.0` and `1.0`. Typical values are in the `0.01-0.2` range.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.
            Values > `1.0` encourage the model to use new tokens, while values < `1.0` encourage the model to repeat
            tokens.
        use_transformers_paged (`bool`, *optional*, defaults to `False`):
            Whether to use the `transformers` paged implementation for generation. If set to `True`, the `transformers`
            paged implementation will be used for generation instead of the default padded implementation. This
            parameter is only effective when `use_vllm` is set to `False`.
        cache_implementation (`str`, *optional*):
            Implementation of the cache method for faster generation when `use_vllm` is set to `False`.
        generation_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to [`~transformers.GenerationConfig`] (if using transformers) or
            `SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the
            generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that conflict
            with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them.

        > Parameters that control generation acceleration powered by vLLM

        use_vllm (`bool`, *optional*, defaults to `False`):
            Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for generation
            instead of the default model.generate(). Requires `vllm` to be installed.
        vllm_mode (`str`, *optional*, defaults to `"server"`):
            Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `"server"` or
            `"colocate"`.

            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
              server is running (start with `trl vllm-serve`).
            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
        vllm_model_impl (`str`, *optional*, defaults to `"vllm"`):
            Model implementation to use for vLLM. Must be one of `"transformers"` or `"vllm"`. `"transformers"`: Use
            the `transformers` backend for model implementation. `"vllm"`: Use the `vllm` library for model
            implementation.
        vllm_guided_decoding_regex (`str`, *optional*):
            Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled.

        > Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)

        vllm_server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `vllm_server_host` and
            `vllm_server_port` are ignored.
        vllm_server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if `vllm_server_base_url` is provided.
        vllm_server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.

        > Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)

        vllm_gpu_memory_utilization (`float`, *optional*, defaults to `0.3`):
            Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_gpu_memory_utilization` flag.
        vllm_tensor_parallel_size (`int`, *optional*, defaults to `1`):
            Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set to
            `"colocate"`. If you are using `vllm_mode="server"`, this parameter must be passed separately when
            launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        vllm_enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Whether to enable sleep mode for vLLM. If `True`, vLLM will sleep during the optimization step and woken
            for weight sync and generation.

        > Parameters that control the training

        beta (`float`, *optional*, defaults to `0.0`):
            KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and improving
            training speed.
        num_iterations (`int`, *optional*, defaults to `1`):
            Number of iterations per batch (denoted as μ in the algorithm).
        epsilon (`float`, *optional*, defaults to `0.2`):
            Epsilon value for clipping.
        delta (`float`, *optional*):
            Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` (default), standard
            GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This method is introduced in
            the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291).
        epsilon_high (`float`, *optional*):
            Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the lower-bound
            specified in argument `epsilon`. Paper [DAPO](https://huggingface.co/papers/2503.14476) recommends `0.28`.
        importance_sampling_level (`str`, *optional*, defaults to `"token"`):
            Controls whether importance sampling ratios are computed at the `"token"` or `"sequence"` level. `"token"`
            keeps the raw per-token log-probability ratios (one weight per token). `"sequence"` averages the
            log-probability ratios across valid tokens to produce a single ratio per sequence. The [GSPO
            paper](https://huggingface.co/papers/2507.18071) shows that sequence-level sampling often yields more
            stable training and better alignment with sequence-level rewards.
        reward_weights (`list[float]`, *optional*):
            Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are
            weighted equally with weight `1.0`.
        scale_rewards (`str` or `bool`, *optional*, defaults to `"group"`):
            Specifies the scaling strategy for rewards. Supported values are:

            - `True` or `"group"` (default): rewards are scaled by the standard deviation within each group, ensuring
              unit variance within a group.
            - `"batch"`: rewards are scaled by the standard deviation across the entire batch, as recommended in the
              [PPO Lite paper](https://huggingface.co/papers/2508.08221).
            - `False` or `"none"`: no scaling is applied. The [Dr. GRPO
              paper](https://huggingface.co/papers/2503.20783) recommends not scaling rewards, as scaling by the
              standard deviation introduces a question-level difficulty bias.
        loss_type (`str`, *optional*, defaults to `"dapo"`):
            Specifies the loss formulation to use. Supported values are:

            - `"grpo"`: Aggregates token-level losses by normalizing over sequence length. Not recommended due to
              length bias—this approach tends to prefer shorter completions with positive advantages and longer ones
              with negative advantages.
            - `"dr_grpo"`: Aggregates token-level losses by normalizing with a global constant. This method was
              introduced in the [Dr. GRPO paper](https://huggingface.co/papers/2503.20783) to eliminate length bias.
              The value of the constant corresponds to `max_completion_length`.
            - `"dapo"` (default): Aggregates token-level losses by normalizing with the number of active token in the
              global accumulated batch. This method was introduced in the [DAPO
              paper](https://huggingface.co/papers/2503.14476) to eliminate length bias.
            - `"bnpo"`: Aggregates token-level losses by normalizing with the number of active token in the local
              batch. Note that normalization is performed over the local batch only, so results may slightly vary
              depending on the local batch size, despite a constant effective batch size. When using
              `per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss.
        mask_truncated_completions (`bool`, *optional*, defaults to `False`):
            When enabled, truncated completions are excluded from the loss calculation, preventing them from being
            incorrectly penalized and introducing noise during training. According to the
            [DAPO](https://huggingface.co/papers/2503.14476) paper, this is a good practice for training stability.
        sync_ref_model (`bool`, *optional*, defaults to `False`):
            Whether to synchronize the reference model with the active model every `ref_model_sync_steps` steps, using
            the `ref_model_mixup_alpha` parameter. This synchronization originates from the
            [TR-DPO](https://huggingface.co/papers/2404.09656) paper.
        ref_model_mixup_alpha (`float`, *optional*, defaults to `0.6`):
            α parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which controls the mix
            between the current policy and the previous reference policy during updates. The reference policy is
            updated according to the equation: `π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you
            must set `sync_ref_model=True`.
        ref_model_sync_steps (`int`, *optional*, defaults to `512`):
            τ parameter from the [TR-DPO](https://huggingface.co/papers/2404.09656) paper, which determines how
            frequently the current policy is synchronized with the reference policy. To use this parameter, you must
            set `sync_ref_model=True`.
        top_entropy_quantile (`float`, *optional*, defaults to `1.0`):
            ρ parameter from [Beyond the 80/20 Rule](https://huggingface.co/papers/2506.01939). Keeps in the policy
            loss term only the top-ρ quantile of tokens by entropy of the probability distribution at each sequence
            position, improving results. Range: `[0.0-1.0]`. A value of `0.0` masks all but the highest entropy token;
            `1.0` keeps all tokens. The paper recommends a value of `0.2`. If used with
            `mask_truncated_completions=True`, only tokens from non-truncated completions are considered.
        use_liger_loss (`bool`, *optional*, defaults to `False`):
            Whether to use the Liger GRPO loss.
        vllm_importance_sampling_correction (`bool`, *optional*, defaults to `True`):
            Whether to apply Truncated Importance Sampling (TIS) between vLLM completion logprobs and recomputed
            logprobs. [Your Efficient RL Framework Secretly Brings You Off-Policy RL
            Training](https://fengyao.notion.site/off-policy-rl) highlights that using a separate generation framework
            (such as vLLM) can introduce off-policy effects due to subtle implementation differences between generation
            and training backends. TIS is proposed as a remedy for this issue.
        vllm_importance_sampling_cap (`float`, *optional*, defaults to `2.0`):
            Truncation parameter C for Truncated Importance Sampling (TIS). This sets an upper bound on the importance
            sampling ratio, improving training stability.

        > Parameters that control the logging

        log_completions (`bool`, *optional*, defaults to `False`):
            Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is installed,
            it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`.
        num_completions_to_print (`int`, *optional*):
            Number of completions to print with `rich`. If `None`, all completions are logged.
        wandb_log_unique_prompts (`bool`, *optional*, defaults to `False`):
            Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, all prompts
            are logged.
    """

    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    # Parameters whose default values are overridden from TrainingArguments
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    logging_steps: float = field(
        default=10,
        metadata={
            "help": "Log every X updates steps. Should be an integer or a float in range `[0,1)`. If smaller than 1, "
            "will be interpreted as ratio of total training steps."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    bf16: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA "
            "architecture or Intel XPU or using CPU (use_cpu) or Ascend NPU. If not set, it defaults to `True` if "
            "`fp16` is not set."
        },
    )

    # Parameters that control the model and reference model
    model_init_kwargs: Optional[Union[dict, str]] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )

    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )
    shuffle_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )

    # Parameters that control generation
    generation_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * steps_per_generation`."
        },
    )
    steps_per_generation: Optional[int] = field(
        default=None,
        metadata={"help": "Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    teacher_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature for teacher logit scaling. If None, uses the same value as `temperature`."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    generation_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or "
            "`SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the "
            "generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that "
            "conflict with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    use_transformers_paged: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the `transformers` paged implementation for generation. If set to `True`, the "
            "`transformers` paged implementation will be used for generation instead of the default padded "
            "implementation. This parameter is only effective when `use_vllm` is set to `False`."
        },
    )
    cache_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    vllm_mode: str = field(
        default="server",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure "
            "a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: "
            "Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for "
            "model implementation."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable sleep mode for vLLM. If `True`, vLLM will sleep during the optimization step "
            "and woken for weight sync and generation."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

    # Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)
    vllm_server_base_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` "
            "and `vllm_server_port` are ignored."
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )

    # Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
        },
    )

    # Parameters that control the training
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and "
            "improving training speed."
        },
    )
    alpha: float = field(
        default=0.0,
        metadata={
            "help": "Alpha coefficient. If `0.0` (default), the forward KL is used. If `1.0`, the reverse KL is used. If anything in between, the Jensen-Shannon Divergence is used."
        },
    )
    generate_from_teacher: bool = field(
        default=False,
        metadata={
            "help": "If True, use the teacher model (ref_model) for generation. vLLM will be initialized with teacher "
                   "weights, enabling fast generation from the teacher. This makes training equivalent to online SFT "
                   "where the teacher generates completions and the student learns to reproduce them. "
                   "If False (default), use the student model for generation (standard RL behavior)."
        },
    )
    splice_generation: bool = field(
        default=False,
        metadata={
            "help": "Enable chimeric two-pass splice generation. The 'starter' role (student or teacher) generates "
                   "a full answer; the first splice_k fraction of its tokens is used as a prefix, and the other "
                   "role continues from that prefix. The stitched completion (prefix + continuation) is used for "
                   "the KL loss over the entire sequence. vLLM holds one set of weights; the starter vs other-role "
                   "distinction is purely which prompt is fed in (student_prompt vs teacher_prompt). Requires "
                   "use_vllm=True. Incompatible with speculative_generation, use_critique, use_onpolicy_demos."
        },
    )
    splice_starter: str = field(
        default="student",
        metadata={
            "help": "Which role runs Pass 1 of the splice. Must be 'student' or 'teacher'. Only active when "
                   "splice_generation=True."
        },
    )
    splice_k: float = field(
        default=0.5,
        metadata={
            "help": "Fraction of Pass 1 answer length used as the prefix for Pass 2. Must be in (0.0, 1.0). "
                   "Only active when splice_generation=True."
        },
    )
    speculative_generation: bool = field(
        default=False,
        metadata={
            "help": "Enable speculative knowledge distillation (SKD) for generation. The student proposes blocks "
                   "of tokens and the teacher verifies via top-K acceptance, replacing rejected tokens. Produces a "
                   "hybrid student/teacher distribution. Requires use_vllm=False. Incompatible with generate_from_teacher."
        },
    )
    speculative_block_size: int = field(
        default=5,
        metadata={
            "help": "Number of tokens (gamma) the student proposes per speculative block before teacher verification. "
                   "Higher values amortize teacher cost but waste more student compute on rejection."
        },
    )
    speculative_acceptance_top_k: int = field(
        default=25,
        metadata={
            "help": "Top-K threshold for teacher acceptance in speculative generation. A student token is accepted if "
                   "it falls within the teacher's top-K predictions. Higher K = more permissive (more student tokens "
                   "accepted); lower K = more teacher corrections."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    delta: Optional[float] = field(
        default=None,
        metadata={
            "help": "Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` "
            "(default), standard GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This "
            "method is introduced in the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)."
        },
    )
    epsilon_high: Optional[float] = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. "
            "`'token'` keeps the raw per-token log-probability ratios (one weight per token).  `'sequence'` averages "
            "the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper "
            "shows that sequence-level sampling often yields more stable training and better alignment with "
            "sequence-level rewards."
        },
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    scale_rewards: str = field(
        default="group",
        metadata={
            "help": "Specifies the scaling strategy for rewards. Supported values are: "
            "`True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring "
            "unit variance within a group. "
            "`'batch'`: rewards are scaled by the standard deviation across the entire batch, as recommended in the "
            "PPO Lite paper. "
            "`False` or `'none'`: no scaling is applied. The Dr. GRPO paper recommends not scaling rewards, as "
            "scaling by the standard deviation introduces a question-level difficulty bias."
        },
    )
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are 'grpo', 'dapo', 'bnpo', and "
            "'dr_grpo'. "
            "'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length "
            "bias—this approach tends to prefer shorter completions with positive advantages and longer ones with "
            "negative advantages. "
            "'dapo' (default): Aggregates token-level losses by normalizing with the number of active token in the "
            "global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias. "
            "'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`. "
            "'bnpo': Aggregates token-level losses by normalizing with the number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss."
        },
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={
            "help": "ρ parameter from Beyond the 80/20 Rule. Keeps in the policy loss term only the top-ρ quantile of "
            "tokens by entropy of the probability distribution at each sequence position, improving results. Range: "
            "[0.0-1.0]. A value of `0.0` masks all but the highest entropy token; `1.0` keeps all tokens. The paper "
            "recommends a value of `0.2`. If used with `mask_truncated_completions=True`, only tokens from "
            "non-truncated completions are considered."
        },
    )
    num_loss_tokens_to_skip: int = field(
        default=0,
        metadata={
            "help": "Number of tokens at the beginning of each completion to exclude from the loss calculation. "
            "This can be useful to avoid penalizing the model for the initial tokens of the response, which may be "
            "less predictable. A value of `0` (default) means all completion tokens are included in the loss."
        },
    )
    force_thinking_prefix: str = field(
        default="",
        metadata={
            "help": "If non-empty, this string is appended to all prompt texts after the chat template is applied, "
            "forcing the model to begin completions with this prefix (e.g. '<think>\\n'). "
            "Default empty means no forced prefix."
        },
    )
    enable_thinking: str = field(
        default="default",
        metadata={
            "help": "Control thinking mode passed to the chat template during training. "
            "'true' = enable thinking, 'false' = disable thinking, "
            "'default' = don't pass enable_thinking (uses tokenizer default, i.e. thinking on for Qwen3)."
        },
    )
    vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply Truncated Importance Sampling (TIS) between vLLM completion logprobs and "
            "recomputed logprobs. Your Efficient RL Framework Secretly Brings You Off-Policy RL "
            "Training highlights that using a separate generation framework (such as vLLM) can introduce off-policy "
            "effects due to subtle implementation differences between generation and training backends. TIS is "
            "proposed as a remedy for this issue."
        },
    )
    vllm_importance_sampling_cap: float = field(
        default=2.0,
        metadata={
            "help": "Truncation parameter C for Truncated Importance Sampling (TIS). This sets an upper bound on the "
            "importance sampling ratio, improving training stability."
        },
    )

    # Parameters that control on-policy demonstrations
    use_onpolicy_demos: bool = field(
        default=False,
        metadata={
            "help": "When True, correct on-policy completions are used as the teacher's in-context demonstration "
            "instead of gold SFT answers. Completions are scored by a reward function after generation; if any "
            "completion for a prompt meets the reward threshold, it replaces the gold demo in the teacher prompt. "
            "Incompatible with speculative_generation and generate_from_teacher."
        },
    )
    onpolicy_demo_reward_threshold: float = field(
        default=1.0,
        metadata={
            "help": "Minimum reward score for a completion to be considered correct and used as an on-policy "
            "demonstration. Only active when use_onpolicy_demos=True."
        },
    )
    strip_thinking_from_demo: bool = field(
        default=True,
        metadata={
            "help": "When True, strip <think>...</think> blocks from on-policy completions before inserting them "
            "as demonstrations in the teacher prompt. Prevents long reasoning traces from consuming the teacher's "
            "prompt length budget. Only active when use_onpolicy_demos=True."
        },
    )

    loss_max_completion_tokens: int = field(
        default=0,
        metadata={
            "help": "Compute the distillation loss on only the first N tokens of each rollout "
            "(0 = all generated tokens, the paper's behaviour). Decouples loss length from "
            "generation length: generation must be long enough for a correctness gate to read a "
            "final answer, but each loss position costs a vocab-wide (B, T, V) fp32 tensor. Since "
            "generation is autoregressive, the first N tokens of a long rollout are distributed "
            "identically to an N-capped rollout, so N=4096 reproduces the paper's loss-token "
            "budget exactly while still allowing max_completion_length=16384 for grading."
        },
    )
    jsd_chunk_size: int = field(
        default=0,
        metadata={
            "help": "Evaluate the per-token JSD in slices of this many sequence positions, with "
            "activation checkpointing per slice (0 = off, single-shot as before). The divergence "
            "block materializes vocab-sized (B, T, V) fp32 tensors only to reduce over V; at "
            "V~152k those are ~9 GiB per tensor at T=16384, which OOMs an 80GB H100. Chunking is "
            "mathematically exact (no top-k truncation) and cuts the peak by roughly T/chunk. "
            "2048 is a good default for long-completion runs."
        },
    )

    # Parameters that control wrong-rollout gating (contrastive OPSD)
    gate_mode: str = field(
        default="none",
        metadata={
            "help": "'none' (default) or 'wrong_only'. When 'wrong_only', the JSD loss is applied only to "
            "rollouts whose final answer is incorrect (and, when gate_require_diff_answer=True, also differs "
            "from the known wrong answer in the 'wrong_answer' dataset column). Rollouts failing the gate are "
            "regenerated up to gate_max_regen_rounds times; rows still failing are zero-masked and excluded "
            "from the loss normalization, so each step averages over live (gated) rollouts only. Requires "
            "'gold_answer' (and optionally 'wrong_answer') dataset columns. Only supported with colocated vLLM "
            "at tensor_parallel_size 1 or non-vLLM generation; incompatible with speculative_generation, "
            "splice_generation, generate_from_teacher, and use_critique."
        },
    )
    gate_max_regen_rounds: int = field(
        default=3,
        metadata={
            "help": "Maximum full-batch regeneration rounds used to replace rollouts that fail the gate. Every "
            "rank always runs the same number of rounds (uniform collectives); regenerated candidates replace "
            "failed rows only. Only active when gate_mode='wrong_only'."
        },
    )
    gate_require_diff_answer: bool = field(
        default=True,
        metadata={
            "help": "When True, gated rollouts must produce a wrong answer that also differs (math-equivalence) "
            "from the teacher-context wrong answer in the 'wrong_answer' column. Only active when "
            "gate_mode='wrong_only'."
        },
    )

    # Parameters that control token-weighted distillation (token-weighted contrastive OPSD)
    token_weight_mode: str = field(
        default="none",
        metadata={
            "help": "'none' (default, exact pre-patch behaviour), 'boxed_hybrid', or 'numeric-skeleton'. When "
            "'boxed_hybrid', the per-token JSD is reweighted per rollout: token_weight_span on the tokens of the "
            "rollout's LAST \\boxed{...} span, token_weight_pre_span on the token_weight_pre_span_tokens tokens "
            "preceding it, and token_weight_epsilon elsewhere. When 'numeric-skeleton' (recommended), the weights "
            "are token_weight_span on the boxed span, token_weight_mid on the result token(s) of EVERY "
            "intermediate arithmetic statement 'a op b = c' in the trace (the group-4 span of the "
            "rl/eval_midtrace_slip.py EQ_RE regex family), and token_weight_epsilon elsewhere; the boxed-span "
            "weight wins where the two overlap. The per-rollout loss is the weight-normalized mean, so every "
            "rollout contributes at the same scale regardless of trace length or weight profile. Rollouts with "
            "no boxed span get all-zero weight (dropped from the loss exactly like gate failures) and are "
            "counted in the weights/no_span_fraction metric. Motivated by the 2026-08-30 measurements: the "
            "pair-specific correctness signal at the wrong trace's own answer slot is +0.84-0.92 nats (27-29 "
            "sigma, ~91% positive; rl/eval_sparse_signal.py), and at the first FALSE mid-trace arithmetic "
            "statement the pair boosts the TRUE result +1.39 nats (31 sigma, 92.5% positive; "
            "rl/eval_midtrace_slip.py) -- while uniform weighting buries both under pair-conditioned style KL "
            "spread over the trace (~100:1). Requires loss_max_completion_tokens=0: the boxed span sits at the "
            "trace END (median ~10.9k tokens on the Arm-1 data), so a first-N loss window would silently "
            "truncate it out of the loss."
        },
    )
    token_weight_epsilon: float = field(
        default=0.001,
        metadata={
            "help": "Weight on tokens outside all weighted spans. 0 disables off-span distillation entirely; "
            "small positive values keep a bounded amount of the paper's full-trace distillation as a hedge. "
            "Active in both 'boxed_hybrid' and 'numeric-skeleton' modes."
        },
    )
    token_weight_span: float = field(
        default=1.0,
        metadata={
            "help": "Weight on the tokens of the rollout's last \\boxed{...} span (opener through matching "
            "closing brace). Active in both 'boxed_hybrid' and 'numeric-skeleton' modes; wins over "
            "token_weight_mid where a result span overlaps the box."
        },
    )
    token_weight_mid: float = field(
        default=0.2,
        metadata={
            "help": "Weight on every intermediate computation-result token -- the stated result c in any "
            "'a op b = c' statement in the trace (true or false; weighting true results too also reinforces "
            "correct-arithmetic conditionals). Only active when token_weight_mode='numeric-skeleton'."
        },
    )
    token_weight_pre_span: float = field(
        default=0.05,
        metadata={
            "help": "Weight on the token_weight_pre_span_tokens tokens immediately before the boxed span (the "
            "final-answer-statement region). Only active when token_weight_mode='boxed_hybrid'."
        },
    )
    token_weight_pre_span_tokens: int = field(
        default=256,
        metadata={
            "help": "Length in tokens of the moderate-weight window before the boxed span. 0 reduces "
            "'boxed_hybrid' to the pure hard-window profile (a). Only active when "
            "token_weight_mode='boxed_hybrid'."
        },
    )

    # Parameters that control critique-conditioned self-teaching
    use_critique: bool = field(
        default=False,
        metadata={
            "help": "When True, the self-teacher is conditioned on a structured critique of the student's rollout "
            "in addition to the gold answer. After student generation, incorrect rollouts receive a generated "
            "critique; correct rollouts receive a canned 'correct' string. The teacher prompt is rebuilt using "
            "teacher_critique_template with the critique. Requires a reward function (--demo_reward_function) "
            "to determine correctness. Incompatible with speculative_generation and generate_from_teacher."
        },
    )
    critique_max_prompt_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum prompt length (in tokens) for the critique generation call. Must be large enough to "
            "fit the problem text, gold solution, and full student rollout. Only active when use_critique=True."
        },
    )
    critique_default_correct: str = field(
        default="The student's solution is correct. Their reasoning successfully arrives at the answer.",
        metadata={
            "help": "Canned critique string used when the student's rollout is correct. Avoids generating a "
            "critique for correct rollouts. Only active when use_critique=True."
        },
    )
    critique_prompt_template: str = field(
        default=(
            "The original task given to the student was:\n\n"
            "[BEGIN_TASK]\n{problem_text}\n[END_TASK]\n\n"
            "[BEGIN_CORRECT_SOLUTION]\n{gold_solution_output}\n[END_CORRECT_SOLUTION]\n\n"
            "A student attempted this problem. Their full reasoning trace is below:\n\n"
            "[BEGIN_STUDENT_REASONING]\n{student_thinking}\n[END_STUDENT_REASONING]\n\n"
            "Analyze the student's problem-solving approach. For each distinct strategy or idea the student tried:\n"
            "1. Briefly describe the approach\n"
            "2. Classify it as PRODUCTIVE, UNPRODUCTIVE, or NEUTRAL\n"
            "3. Briefly explain why\n\n"
            "Then, working backwards from the correct solution, explain how the student could have arrived at it "
            "from their existing work. Identify any intermediate results the student already computed that could "
            "have been used differently to reach the answer. Be concise."
        ),
        metadata={
            "help": "Template for generating the critique. Placeholders: {problem_text}, {gold_solution_output}, "
            "{student_thinking}. Only active when use_critique=True."
        },
    )
    teacher_critique_template: str = field(
        default=(
            "{problem}\n\n"
            "Below is an example of a reference solution:\n\n"
            "[BEGIN_REFERENCE_ANSWER]\n{gold_solution_output}\n[END_REFERENCE_ANSWER]\n\n"
            "The following is feedback for a student's earlier attempt:\n\n"
            "[BEGIN_CRITIQUE_OF_STUDENT_ATTEMPT]\n{critique}\n[END_CRITIQUE_OF_STUDENT_ATTEMPT]\n\n"
            "Now, correctly solve this problem, informed by the analysis above."
        ),
        metadata={
            "help": "Template for the teacher prompt with critique in context. Placeholders: {problem}, "
            "{gold_solution_output}, {critique}. Only active when use_critique=True."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    num_completions_to_print: Optional[int] = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )
    wandb_log_unique_prompts: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts in wandb. If `True`, only unique prompts are logged. If `False`, "
            "all prompts are logged."
        },
    )

    def __post_init__(self):
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        super().__post_init__()

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % self.num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by num_generations ({self.num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.delta is not None and self.use_liger_loss:
            raise ValueError("Liger loss does not support two-sided GRPO loss yet.")

        if self.speculative_generation:
            if self.generate_from_teacher:
                raise ValueError(
                    "speculative_generation is incompatible with generate_from_teacher. In speculative mode, "
                    "both student and teacher cooperate during generation."
                )
            if self.use_vllm:
                raise ValueError(
                    "speculative_generation requires use_vllm=False. Speculative generation uses a custom "
                    "token-by-token loop with both models and cannot delegate to vLLM."
                )
            if self.speculative_block_size < 1:
                raise ValueError(f"speculative_block_size must be >= 1, got {self.speculative_block_size}")
            if self.speculative_acceptance_top_k < 1:
                raise ValueError(f"speculative_acceptance_top_k must be >= 1, got {self.speculative_acceptance_top_k}")
            if self.per_device_train_batch_size > 1:
                raise ValueError(
                    "speculative_generation requires per_device_train_batch_size=1. The batched KV cache "
                    "forces a batch-minimum acceptance count, which corrupts per-sequence speculative "
                    "decoding when batch size > 1."
                )
            if self.min_p is not None and self.min_p > 0.0:
                raise ValueError(
                    "min_p filtering is not implemented in the speculative generation loop. "
                    f"Got min_p={self.min_p}; set min_p=None or 0.0 when using speculative_generation."
                )
            if self.repetition_penalty != 1.0:
                raise ValueError(
                    "repetition_penalty is not implemented in the speculative generation loop. "
                    f"Got repetition_penalty={self.repetition_penalty}; set repetition_penalty=1.0 "
                    "when using speculative_generation."
                )

        if self.use_onpolicy_demos:
            if self.speculative_generation:
                raise ValueError(
                    "use_onpolicy_demos is incompatible with speculative_generation. Speculative generation "
                    "requires teacher prompts before generation, but on-policy demos are constructed after."
                )
            if self.generate_from_teacher:
                raise ValueError(
                    "use_onpolicy_demos is incompatible with generate_from_teacher. On-policy demos require "
                    "student-generated completions; teacher-generated completions are not on-policy."
                )

        if self.use_critique:
            if self.speculative_generation:
                raise ValueError(
                    "use_critique is incompatible with speculative_generation. Speculative generation "
                    "requires teacher prompts before generation, but critique-conditioned prompts are "
                    "constructed after student rollout."
                )
            if self.generate_from_teacher:
                raise ValueError(
                    "use_critique is incompatible with generate_from_teacher. Critique requires "
                    "student-generated completions to analyze; teacher-generated completions are not on-policy."
                )

        if self.splice_generation:
            if self.splice_starter not in {"student", "teacher"}:
                raise ValueError(
                    f"splice_starter must be 'student' or 'teacher', got {self.splice_starter!r}."
                )
            if not (0.0 < self.splice_k < 1.0):
                raise ValueError(
                    f"splice_k must be in (0.0, 1.0), got {self.splice_k}."
                )
            if not self.use_vllm:
                raise ValueError(
                    "splice_generation requires use_vllm=True. The two-pass splice is implemented on top of "
                    "the vLLM generation path."
                )
            if self.vllm_mode != "colocate" and self.num_generations > 1:
                raise ValueError(
                    "splice_generation with num_generations>1 requires vllm_mode='colocate'. The server-mode "
                    "dedupe-and-expand path generates one output per unique prompt, but splice needs a distinct "
                    "continuation prefix per sample."
                )
            if self.max_prompt_length is not None and self.max_prompt_length < self.max_completion_length:
                raise ValueError(
                    f"splice_generation requires max_prompt_length >= max_completion_length, got "
                    f"max_prompt_length={self.max_prompt_length}, max_completion_length={self.max_completion_length}. "
                    f"Pass 2 feeds the decoded prefix (up to splice_k * max_completion_length tokens) into the "
                    f"prompt; if max_prompt_length is too small, vLLM will silently truncate the prefix."
                )
            if self.speculative_generation:
                raise ValueError(
                    "splice_generation is incompatible with speculative_generation."
                )
            if self.use_critique:
                raise ValueError(
                    "splice_generation is incompatible with use_critique."
                )
            if self.use_onpolicy_demos:
                raise ValueError(
                    "splice_generation is incompatible with use_onpolicy_demos."
                )