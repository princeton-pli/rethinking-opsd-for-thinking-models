# adapted from https://github.com/idanshen/Self-Distillation

import inspect
import os
import re

# Regex patterns for stripping thinking tokens across model families
_THINKING_PATTERNS = [
    re.compile(r'<think>[\s\S]*?</think>'),           # Qwen3
    re.compile(r'<\|channel>thought[\s\S]*?<channel\|>'),  # Gemma 4
]


def strip_thinking_tokens(text: str) -> str:
    for pat in _THINKING_PATTERNS:
        text = pat.sub('', text)
    return text.strip()
import textwrap
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from string import Template
from typing import Any, Callable, Optional, Union

import datasets
import torch
import torch.utils.data
import transformers
from accelerate import logging
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from datasets import Dataset, IterableDataset
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Sampler
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_wandb_available,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available, is_flash_attn_2_available, is_peft_available, is_rich_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template, prepare_multimodal_messages
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from trl.import_utils import is_liger_kernel_available, is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, prepare_peft_model, unwrap_model_for_generation
from trl.models.utils import _ForwardRedirection
from trl.trainer.base_trainer import BaseTrainer
from .config import DistilConfig
from accelerate.state import AcceleratorState
from trl.trainer.utils import (
    RepeatSampler,
    disable_dropout_in_model,
    ensure_master_addr_port,
    entropy_from_logits,
    identity,
    nanmax,
    nanmin,
    nanstd,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
    shuffle_sequence_dict,
    split_pixel_values_by_grid,
    split_tensor_dict,
    unsplit_pixel_values_by_grid,
)
from torch.nn.functional import log_softmax, kl_div


if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb


logger = logging.get_logger(__name__)


class MemoryEfficientSyncRefModelCallback(TrainerCallback):
    """
    Memory-efficient callback to synchronize the model with a reference model.
    
    Unlike the default SyncRefModelCallback, this version iterates through parameters
    one at a time instead of gathering all parameters at once. This reduces peak memory
    usage from O(full_model_size) to O(single_param_size), making it feasible to sync
    large models with DeepSpeed ZeRO-3.
    """

    def __init__(
        self,
        ref_model: Union[PreTrainedModel, nn.Module],
        accelerator: Optional[Any],
    ):
        self.accelerator = accelerator
        self.ref_model = ref_model

    @staticmethod
    def _sync_param(model_param, ref_param, alpha):
        """Sync a single parameter: ref = alpha * model + (1 - alpha) * ref"""
        ref_param.data.mul_(1.0 - alpha).add_(model_param.data, alpha=alpha)

    @staticmethod
    def sync_target_model_memory_efficient(model, target_model, alpha):
        """
        Sync target_model to track model, gathering one parameter at a time.
        
        This is O(1) in memory overhead instead of O(N) where N is model size.
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin
        is_zero3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        
        if is_zero3:
            import deepspeed
            
            # Iterate through parameters one at a time
            for (name, model_param), (_, ref_param) in zip(
                model.named_parameters(), target_model.named_parameters()
            ):
                # Gather only this pair of parameters
                with deepspeed.zero.GatheredParameters(
                    [model_param, ref_param], modifier_rank=0
                ):
                    if deepspeed.comm.get_rank() == 0:
                        MemoryEfficientSyncRefModelCallback._sync_param(
                            model_param, ref_param, alpha
                        )
        else:
            # Non-ZeRO-3: just iterate normally
            for model_param, ref_param in zip(model.parameters(), target_model.parameters()):
                MemoryEfficientSyncRefModelCallback._sync_param(model_param, ref_param, alpha)

    def on_step_end(self, args, state, control, **kwargs):
        model: PreTrainedModel = kwargs["model"]

        if self.ref_model is not None and state.global_step % args.ref_model_sync_steps == 0:
            if self.accelerator:
                model = self.accelerator.unwrap_model(model)
            self.sync_target_model_memory_efficient(model, self.ref_model, args.ref_model_mixup_alpha)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class DistilTrainer(BaseTrainer):
    """
    Trainer for the Self-Distillation method. 

    Example:

    ```python
    from datasets import load_dataset
    from trl import DistilTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")


    def reward_func(completions, **kwargs):
        # Dummy reward function that rewards completions with more unique letters.
        return [float(len(set(completion))) for completion in completions]


    trainer = DistilTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_func,
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or a
              path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
              using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keyword arguments in
              `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. Custom reward
                  functions can also return `None` when the reward is not applicable to those samples. This is useful
                  for multi-task training where different reward functions apply to different types of samples. When a
                  reward function returns `None` for a sample, that reward function is excluded from the reward
                  calculation for that sample. For more details, see [Using a custom reward
                  function](#using-a-custom-reward-function).

                  The trainer's state is also passed to the reward function. The trainer's state is an instance of
                  [`~transformers.TrainerState`] and can be accessed by accessing the `trainer_state` argument to the
                  reward function's signature.
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`DistilConfig`], *optional*):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], [`~transformers.ProcessorMixin`], *optional*):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoProcessor.from_pretrained`]. A
            padding token, `tokenizer.pad_token`, must be set. If the processing class has not set a padding token,
            `tokenizer.eos_token` will be used as the default.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks detailed
            in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "distil"]
    _name = "Distil"

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        ref_model: Union[str, PreTrainedModel],
        args: Optional[DistilConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[Union[PreTrainedTokenizerBase, ProcessorMixin]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = DistilConfig(f"{model_name}-Distil")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            dtype = model_init_kwargs.get("dtype")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass  # dtype is already a torch.dtype or "auto" or None
            elif isinstance(dtype, str):  # it's a str, but not "auto"
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `DistilConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            model = architecture.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `DistilConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        if peft_config is not None or (is_peft_available() and isinstance(model, PeftModel)):
            model = prepare_peft_model(model, peft_config, args)

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model.config._name_or_path, truncation_side="left")

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.temperature = args.temperature
        self.teacher_temperature = args.teacher_temperature if args.teacher_temperature is not None else args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_transformers_paged = args.use_transformers_paged
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        self.num_loss_tokens_to_skip = args.num_loss_tokens_to_skip

        # Speculative generation (SKD)
        self.speculative_generation = args.speculative_generation
        self.speculative_block_size = args.speculative_block_size
        self.speculative_top_k = args.speculative_acceptance_top_k

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in DistilTrainer. Please use a standard dataset instead."
            )

        # Multi-step
        self.num_iterations = args.num_iterations
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO-like algorithms, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        # model.warnings_issued["estimate_tokens"] = True  # not available on Gemma4ForConditionalGeneration
        if hasattr(model, "warnings_issued"):
            model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in Distil
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )

        # Reference model
        self.beta = args.beta
        self.alpha = args.alpha
        self.generate_from_teacher = args.generate_from_teacher
        if ref_model is not None:
            # If a reference model is provided, use it
            self.ref_model = ref_model
        elif self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "images": deque(maxlen=args.generation_batch_size),
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
        }

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                    self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                    self.vllm_client.init_communicator(device=torch.cuda.current_device())

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                if self.max_prompt_length is not None and self.max_completion_length is not None:
                    max_model_len = self.max_prompt_length + self.max_completion_length
                else:
                    max_model_len = None
                # Use teacher model for vLLM when generate_from_teacher=True
                vllm_model_path = ref_model.name_or_path if self.generate_from_teacher and ref_model is not None else model.name_or_path
                self.llm = LLM(
                    model=vllm_model_path,
                    tensor_parallel_size=args.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size
                    * self.args.steps_per_generation,
                    max_model_len=max_model_len,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                    max_num_batched_tokens=4096,
                    model_impl=self.args.vllm_model_impl,
                    enable_sleep_mode=self.args.vllm_enable_sleep_mode,
                    # Important so temperature scaling/logit tweaking affects the TIS log probs
                    logprobs_mode="processed_logprobs",
                )
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)
            else:
                raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'.")

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": tokenizer.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(MemoryEfficientSyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In DistilTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "teacher_prompt", "image", "images", "gold_answer", "gate_gold_answer", "wrong_answer"]

    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification. As a result, some parts of the method aren't relevant to Distil, but we keep them to stay one line
    # apart from the super method, ensuring easier maintenance in the future.
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )

            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self, dataset: Optional[Dataset] = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        temperature=None,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
        compute_all_logps=True,
    ) -> dict[str, Optional[torch.Tensor]]:
        """Compute log-probs and (optionally) entropies for each token."""
        temperature = temperature if temperature is not None else self.temperature
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_selected_logps = []
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / temperature

            completion_ids = input_ids_batch[:, -logits_to_keep:]
            selected_logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            if compute_all_logps:
                logps = log_softmax(logits, dim=-1)
            else:
                logps = None
            all_selected_logps.append(selected_logps)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        selected_logps = torch.cat(all_selected_logps, dim=0)
        if compute_all_logps:
            logps = torch.cat(all_logps, dim=0)
        else:
            logps = None
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return selected_logps, logps, entropies

    def _fix_param_name_to_vllm(self, name, extra_prefixes: Optional[list[str]] = None):
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    @profiling_decorator
    def _move_model_to_vllm(self):
        # Select which model to sync to vLLM: teacher (ref_model) or student (model)
        # When generate_from_teacher=True, sync the teacher model since vLLM was initialized with teacher weights
        model_to_sync = self.ref_model if self.generate_from_teacher else self.model
        
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(self.model):
            if self.generate_from_teacher:
                raise ValueError("PEFT model handling only applies when syncing student model (teacher is typically not PEFT)")
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(
                            self.model
                        )  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(model_to_sync)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(model_to_sync)
            else:
                for name, param in model_to_sync.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()

    @profiling_decorator
    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = split_pixel_values_by_grid(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [unsplit_pixel_values_by_grid(batch) for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items() if key != "trainer_state"
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    @staticmethod
    def _trim_kv_cache(past_key_values, new_seq_len):
        """Trim a DynamicCache's sequence dimension to new_seq_len."""
        if hasattr(past_key_values, "crop"):
            past_key_values.crop(new_seq_len)
        elif hasattr(past_key_values, "key_cache"):
            for i in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[i] = past_key_values.key_cache[i][:, :, :new_seq_len, :]
                past_key_values.value_cache[i] = past_key_values.value_cache[i][:, :, :new_seq_len, :]
            if hasattr(past_key_values, "_seen_tokens"):
                past_key_values._seen_tokens = new_seq_len
        else:
            raise TypeError(
                f"Cannot trim KV cache of type {type(past_key_values)}. "
                "Expected a DynamicCache with crop() or key_cache/value_cache attributes."
            )

    @staticmethod
    @contextmanager
    def _bypass_fsdp_forward(model):
        """Temporarily bypass FSDP's forward method and hooks on all FSDP submodules.

        FSDP overrides forward() to run all-gather/reshard (NCCL collectives).
        This replaces each FSDP module's forward with its inner module's forward
        and clears all forward hooks, making forward passes purely local.

        Must be used inside summon_full_params(recurse=True) which ensures all
        parameters are already gathered on each rank.
        """
        _HOOK_ATTRS = (
            "_forward_pre_hooks",
            "_forward_hooks",
            "_forward_pre_hooks_with_kwargs",
            "_forward_hooks_with_kwargs",
        )
        saved = {}
        for name, mod in model.named_modules():
            if isinstance(mod, FSDP):
                mod_saved = {}
                for attr in _HOOK_ATTRS:
                    hook_dict = getattr(mod, attr, None)
                    if hook_dict is not None:
                        mod_saved[attr] = dict(hook_dict)
                        hook_dict.clear()
                inner = getattr(mod, '_fsdp_wrapped_module', None) or getattr(mod, 'module', None)
                if inner is not None:
                    mod_saved['_orig_forward'] = mod.forward
                    mod.forward = inner.forward
                saved[name] = mod_saved
        try:
            yield
        finally:
            for name, mod in model.named_modules():
                if name in saved:
                    mod_data = saved[name]
                    if '_orig_forward' in mod_data:
                        mod.forward = mod_data['_orig_forward']
                    for attr in _HOOK_ATTRS:
                        if attr in mod_data:
                            getattr(mod, attr).update(mod_data[attr])

    def _speculative_generate(
        self,
        student_prompt_ids: torch.Tensor,
        student_attn_mask: torch.Tensor,
        teacher_prompt_ids: torch.Tensor,
        teacher_attn_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """
        Block-based speculative knowledge distillation generation (Xu et al., 2024).

        The student proposes blocks of gamma tokens; the teacher verifies all gamma tokens
        in a single forward pass using top-K acceptance. On the first rejection, the remaining
        proposed tokens are discarded, the rejected token is replaced with a sample from the
        teacher's distribution, and the student resumes from the corrected prefix. When all
        gamma tokens are accepted, generation continues with the next block.

        Both models maintain independent KV caches (the teacher has a longer prompt due to
        privileged context). When FSDP is enabled, full parameters are gathered via
        summon_full_params(recurse=True) so each rank can run generation independently.

        Returns:
            completion_ids: (B, L) tensor of generated token ids (right-padded with pad_token_id)
            acceptance_rate: fraction of proposed student tokens that were accepted
        """
        device = student_prompt_ids.device
        B = student_prompt_ids.shape[0]
        gamma = self.speculative_block_size
        K = self.speculative_top_k
        temp = self.temperature
        max_len = self.max_completion_length
        pad_id = self.pad_token_id
        eos_id = self.eos_token_id

        unwrapped_student = self.accelerator.unwrap_model(self.model)
        unwrapped_teacher = self.accelerator.unwrap_model(self.ref_model)

        prev_s_cache = getattr(unwrapped_student.config, "use_cache", False)
        prev_t_cache = getattr(unwrapped_teacher.config, "use_cache", False)
        unwrapped_student.config.use_cache = True
        unwrapped_teacher.config.use_cache = True

        s_had_grad_ckpt = getattr(unwrapped_student, "is_gradient_checkpointing", False)
        t_had_grad_ckpt = getattr(unwrapped_teacher, "is_gradient_checkpointing", False)
        if s_had_grad_ckpt and hasattr(unwrapped_student, "gradient_checkpointing_disable"):
            unwrapped_student.gradient_checkpointing_disable()
        if t_had_grad_ckpt and hasattr(unwrapped_teacher, "gradient_checkpointing_disable"):
            unwrapped_teacher.gradient_checkpointing_disable()

        try:
            fsdp_ctx_student = (
                FSDP.summon_full_params(self.model_wrapped, recurse=True)
                if self.is_fsdp_enabled else nullcontext()
            )
            fsdp_ctx_teacher = (
                FSDP.summon_full_params(self.ref_model, recurse=True)
                if self.is_fsdp_enabled and isinstance(self.ref_model, FSDP) else nullcontext()
            )

            disable_hooks_student = (
                self._bypass_fsdp_forward(self.model_wrapped)
                if self.is_fsdp_enabled else nullcontext()
            )
            disable_hooks_teacher = (
                self._bypass_fsdp_forward(self.ref_model)
                if self.is_fsdp_enabled and isinstance(self.ref_model, FSDP) else nullcontext()
            )

            with (
                torch.no_grad(),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as _,
                fsdp_ctx_student,
                fsdp_ctx_teacher,
                disable_hooks_student,
                disable_hooks_teacher,
            ):
                ones_col = torch.ones(B, 1, dtype=student_attn_mask.dtype, device=device)

                # === PREFILL both models ===
                s_out = unwrapped_student(student_prompt_ids, attention_mask=student_attn_mask)
                s_kv = s_out.past_key_values
                s_logits = s_out.logits[:, -1, :]  # raw logits for first completion position

                t_out = unwrapped_teacher(teacher_prompt_ids, attention_mask=teacher_attn_mask)
                t_kv = t_out.past_key_values
                t_prev_logits = t_out.logits[:, -1, :]  # for verifying the first proposed token

                s_mask = student_attn_mask.clone()
                t_mask = teacher_attn_mask.clone()

                generated = []  # list of (B, 1) token tensors
                finished = torch.zeros(B, dtype=torch.bool, device=device)
                total_proposed = 0
                total_accepted = 0

                while len(generated) < max_len:
                    block_size = min(gamma, max_len - len(generated))

                    # ── PROPOSE: student autoregressively generates block_size tokens ──
                    proposed = []
                    for _ in range(block_size):
                        scaled_logits = s_logits / temp
                        if self.top_k is not None and self.top_k > 0:
                            topk_vals, _ = scaled_logits.topk(self.top_k, dim=-1)
                            scaled_logits = scaled_logits.masked_fill(
                                scaled_logits < topk_vals[:, -1:], -float("inf")
                            )
                        if self.top_p is not None and self.top_p < 1.0:
                            sorted_logits, sorted_idx = scaled_logits.sort(dim=-1, descending=True)
                            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                            remove = cumprobs - sorted_logits.softmax(dim=-1) >= self.top_p
                            sorted_logits[remove] = -float("inf")
                            scaled_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                        probs = torch.softmax(scaled_logits, dim=-1)
                        tok = torch.multinomial(probs, num_samples=1)  # (B, 1)
                        tok = torch.where(finished.unsqueeze(-1), pad_id, tok)
                        proposed.append(tok)

                        s_mask = torch.cat([s_mask, ones_col], dim=1)
                        s_out = unwrapped_student(tok, attention_mask=s_mask, past_key_values=s_kv)
                        s_kv = s_out.past_key_values
                        s_logits = s_out.logits[:, -1, :]

                    proposed_block = torch.cat(proposed, dim=1)  # (B, block_size)

                    # ── VERIFY: teacher checks all block_size tokens in one forward pass ──
                    t_mask = torch.cat(
                        [t_mask, torch.ones(B, block_size, dtype=t_mask.dtype, device=device)], dim=1
                    )
                    t_out = unwrapped_teacher(
                        proposed_block, attention_mask=t_mask, past_key_values=t_kv,
                    )
                    # t_out.logits shape: (B, block_size, V)
                    #   t_out.logits[:, j, :] = p_T(· | prefix, proposed[0..j])
                    # For verification we need p_T(· | prefix, proposed[0..j-1]) at position j.
                    # That is t_prev_logits for j=0, and t_out.logits[:, j-1, :] for j>=1.
                    verify_logits = torch.cat(
                        [t_prev_logits.unsqueeze(1), t_out.logits[:, :-1, :]], dim=1
                    ) / temp  # (B, block_size, V)

                    # ── ACCEPT / REJECT (left-to-right) ──
                    _, topk_indices = verify_logits.topk(K, dim=-1)  # (B, block_size, K)
                    in_topk = (topk_indices == proposed_block.unsqueeze(-1)).any(dim=-1)  # (B, block_size)
                    in_topk = in_topk | finished.unsqueeze(-1)  # finished seqs always "accept"

                    # Number of leading accepted tokens per sequence
                    n_accept_per_seq = in_topk.cumprod(dim=-1).sum(dim=-1)  # (B,)
                    # Minimum across batch (KV cache seq_len dimension is shared)
                    n_accept = n_accept_per_seq.min().item()
                    all_accepted = n_accept == block_size

                    total_proposed += block_size
                    total_accepted += n_accept

                    if all_accepted:
                        # All proposed tokens accepted — commit them
                        for tok in proposed:
                            generated.append(tok)
                            finished = finished | (tok.squeeze(-1) == eos_id)

                        # Teacher KV and mask are already up to date from verification
                        t_kv = t_out.past_key_values
                        t_prev_logits = t_out.logits[:, -1, :]

                    else:
                        # Partial acceptance — commit accepted tokens, then corrected token
                        for j in range(n_accept):
                            generated.append(proposed[j])
                            finished = finished | (proposed[j].squeeze(-1) == eos_id)

                        # Sample corrected token from teacher's full distribution
                        rej_logits = verify_logits[:, n_accept, :]  # (B, V) — already /temp
                        corrected = torch.multinomial(
                            torch.softmax(rej_logits, dim=-1), 1
                        )  # (B, 1)
                        corrected = torch.where(finished.unsqueeze(-1), pad_id, corrected)
                        generated.append(corrected)
                        finished = finished | (corrected.squeeze(-1) == eos_id)

                        # Rollback student KV cache: discard entries for proposed[n_accept:]
                        s_rollback_len = s_mask.shape[1] - (block_size - n_accept)
                        self._trim_kv_cache(s_kv, s_rollback_len)
                        s_mask = s_mask[:, :s_rollback_len]

                        # Rollback teacher KV cache: discard entries for proposed[n_accept:]
                        t_rollback_len = t_mask.shape[1] - (block_size - n_accept)
                        self._trim_kv_cache(t_out.past_key_values, t_rollback_len)
                        t_mask = t_mask[:, :t_rollback_len]

                        # Process corrected token through both models
                        s_mask = torch.cat([s_mask, ones_col], dim=1)
                        s_out = unwrapped_student(corrected, attention_mask=s_mask, past_key_values=s_kv)
                        s_kv = s_out.past_key_values
                        s_logits = s_out.logits[:, -1, :]

                        t_mask = torch.cat([t_mask, ones_col], dim=1)
                        t_out_corr = unwrapped_teacher(
                            corrected, attention_mask=t_mask,
                            past_key_values=t_out.past_key_values,
                        )
                        t_kv = t_out_corr.past_key_values
                        t_prev_logits = t_out_corr.logits[:, -1, :]

                    if finished.all():
                        break

                # Sync all ranks before exiting summon_full_params context
                if self.is_fsdp_enabled:
                    torch.distributed.barrier()

                # Assemble output: pad to uniform length
                if generated:
                    completion_ids = torch.cat(generated, dim=1)  # (B, n_generated)
                else:
                    completion_ids = torch.zeros(B, 0, dtype=torch.long, device=device)

                acceptance_rate = total_accepted / max(total_proposed, 1)

        finally:
            unwrapped_student.config.use_cache = prev_s_cache
            unwrapped_teacher.config.use_cache = prev_t_cache
            if s_had_grad_ckpt and hasattr(unwrapped_student, "gradient_checkpointing_enable"):
                unwrapped_student.gradient_checkpointing_enable()
            if t_had_grad_ckpt and hasattr(unwrapped_teacher, "gradient_checkpointing_enable"):
                unwrapped_teacher.gradient_checkpointing_enable()

        return completion_ids, acceptance_rate

    def _generate_single_turn(self, prompts: list[str], images: Optional[list], continuation_prefixes: Optional[list[str]] = None):
        device = self.accelerator.device

        # If the prompts are conversational and the inputs contain images, we need to convert the prompts from
        # [{"role": "user", "content": "What color is the sky?"}] to
        # [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky?"}]}]
        kwargs = {}
        if images is not None:
            kwargs = {"images": images}
            for prompt, image_list in zip(prompts, images):
                if isinstance(prompt, list):  # i.e., when using conversational data
                    prepare_multimodal_messages(prompt, num_images=len(image_list))

        chat_template_kwargs = {}
        if self.args.enable_thinking != "default":
            chat_template_kwargs["enable_thinking"] = (self.args.enable_thinking == "true")

        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt, "chat_template_kwargs": chat_template_kwargs}, self.processing_class)["prompt"] for prompt in prompts
        ]
        if self.args.force_thinking_prefix:
            prompts_text = [p + self.args.force_thinking_prefix for p in prompts_text]
        if continuation_prefixes is not None:
            # Splice path: append a decoded prefix from a previous generation pass to each prompt so the model
            # continues from where the previous pass left off. The prefix stays in the prompt; vLLM returns only
            # the new suffix as the completion.
            assert len(continuation_prefixes) == len(prompts_text), (
                f"continuation_prefixes length {len(continuation_prefixes)} != prompts_text length {len(prompts_text)}"
            )
            prompts_text = [p + cp for p, cp in zip(prompts_text, continuation_prefixes)]

        if images is not None:
            prompt_inputs = self.processing_class(text=prompts_text, padding=True, return_tensors="pt", **kwargs)
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # Generate completions using either vLLM or regular generation
        # Note: When generate_from_teacher=True, vLLM is initialized with teacher weights
        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                # wake up colocated vLLM instances if needed
                torch.cuda.empty_cache()  # required to avoid OOM in some cases
                self.llm.wake_up()

            # First, update the vLLM weights if needed
            # When generate_from_teacher=True and sync_ref_model=False, teacher is static so no sync needed
            # (vLLM already loaded teacher weights at initialization)
            should_sync = self.state.global_step != self._last_loaded_step
            if self.generate_from_teacher and not self.args.sync_ref_model:
                should_sync = False  # Teacher is static, no need to sync
            if should_sync:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)
                if images is not None:
                    all_images = gather_object(images)

                if self.accelerator.is_main_process:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                    if images is not None:
                        ordered_set_of_images = all_images[:: self.num_generations]
                    else:
                        ordered_set_of_images = None

                    with profiling_context(self, "vLLM.generate"):
                        output = self.vllm_client.generate(
                            prompts=ordered_set_of_prompts,
                            images=ordered_set_of_images,
                            n=self.num_generations,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            truncate_prompt_tokens=self.max_prompt_length,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                        payload = (output["prompt_ids"], output["completion_ids"], output["logprobs"])
                else:
                    payload = None

                # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_prompt_ids, all_completion_ids, all_logprobs = obj_list[0]

                # At this point, we only get 1 copy of each prompt, so we need to repeat them num_generations times
                all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(self.num_generations)]

                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                prompt_ids = all_prompt_ids[process_slice]
                completion_ids = all_completion_ids[process_slice]
                logprobs = all_logprobs[process_slice]

            # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
            elif self.vllm_mode == "colocate":
                generation_kwargs = {
                    "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.max_prompt_length,
                    "logprobs": 0,  # only return the logprob of the generated token
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    # Gather prompts from all ranks in the TP group and flatten.
                    # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text = [p for sublist in gathered_prompts for p in sublist]

                    if images is not None:
                        gathered_images = [None for _ in range(self.vllm_tensor_parallel_size)]
                        torch.distributed.all_gather_object(gathered_images, images, group=self.tp_group)
                        all_images = [img for sublist in gathered_images for img in sublist]
                    else:
                        all_images = None
                else:
                    all_prompts_text = prompts_text
                    all_images = images

                if images is not None and all_images:
                    vllm_inputs = []
                    for prompt, image_list in zip(all_prompts_text, all_images):
                        vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_list}})

                else:
                    vllm_inputs = all_prompts_text

                with profiling_context(self, "vLLM.generate"):
                    all_outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params, use_tqdm=False)

                all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
                all_logprobs = [
                    [next(iter(lp.values())).logprob for lp in output.logprobs]
                    for outputs in all_outputs
                    for output in outputs.outputs
                ]

                if self.vllm_tensor_parallel_size > 1:
                    # Slice completions for this rank within its TP group.
                    # Each rank generates all outputs — we keep only our share.
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    prompt_ids = all_prompt_ids[tp_slice]
                    completion_ids = all_completion_ids[tp_slice]
                    logprobs = all_logprobs[tp_slice]
                else:
                    prompt_ids = all_prompt_ids
                    completion_ids = all_completion_ids
                    logprobs = all_logprobs

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

        elif self.use_transformers_paged:
            # Re-process inputs for paged generation if needed
            # Note: images are already validated and preprocessed above
            paged_prompt_inputs = self.processing_class(text=prompts_text, **kwargs)
            previous_attn = self.model_wrapped.config._attn_implementation

            if is_flash_attn_2_available():
                self.model_wrapped.config._attn_implementation = "paged_attention"
            else:
                self.model_wrapped.config._attn_implementation = "sdpa_paged"
            with (
                profiling_context(self, "transformers.generate_batch"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                # Cast to the appropriate dtype based on training configuration
                if self.args.bf16:
                    unwrapped_model.to(torch.bfloat16)
                elif self.args.fp16:
                    unwrapped_model.to(torch.float16)
                with torch.inference_mode():
                    all_outputs = unwrapped_model.generate_batch(
                        paged_prompt_inputs.input_ids, generation_config=self.generation_config, progress_bar=False
                    )
                    unwrapped_model.train()  # restore training mode, as generate_batch forces eval mode
            completion_ids = [output.generated_tokens for output in all_outputs.values()]
            prompt_ids = paged_prompt_inputs.input_ids
            # Restore the original attention implementation, training mode
            self.model_wrapped.config._attn_implementation = previous_attn
            logprobs = None  # not used in this case

        else:
            # Regular generation path
            generate_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                **kwargs,
            )
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config, disable_compile=True
                )
            # Compute prompt length and extract completion ids
            prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Mask everything after the first EOS token
            is_eos = completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            prompt_ids = [p[m].tolist() for p, m in zip(prompt_ids, prompt_mask.bool())]
            completion_ids = [c[m].tolist() for c, m in zip(completion_ids, completion_mask.bool())]
            logprobs = None  # not used in this case

        return prompt_ids, completion_ids, logprobs, forward_kwargs

    def _generate(self, prompts: list[str], images: Optional[list], continuation_prefixes: Optional[list[str]] = None):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompt_ids, completion_ids, logprobs, forward_kwargs = self._generate_single_turn(
            prompts, images, continuation_prefixes=continuation_prefixes
        )

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()  # = num_items_in_batch, required for the DAPO loss

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs

    def _generate_spliced(self, prompts, teacher_prompts, images):
        """Two-pass chimeric splice.

        Pass 1: the starter role (student or teacher) generates a full answer using its own prompt.
        Pass 2: the other role continues from the first splice_k fraction of Pass 1's tokens, using
        its own prompt + the decoded prefix appended to the chat-templated string. The returned
        completion is the stitched (prefix + Pass 2 suffix), truncated to max_completion_length.

        The sampling logprobs are stitched the same way (Pass 1 prefix logprobs + Pass 2 suffix
        logprobs), so the existing vLLM importance-sampling correction in
        _generate_and_score_completions remains meaningful.

        Returns the same 5-tuple shape as _generate:
            (prompt_ids, completion_ids, total_completion_tokens, logprobs, forward_kwargs)
        The returned prompt_ids are from Pass 2 and are discarded by the caller (which re-tokenizes
        prompts and teacher_prompts independently downstream).
        """
        starter = self.args.splice_starter
        k = self.args.splice_k

        pass1_prompts = prompts if starter == "student" else teacher_prompts
        pass2_prompts = teacher_prompts if starter == "student" else prompts

        # Pass 1: starter generates full answer
        _p1_prompt_ids, p1_completion_ids, _p1_total, p1_logprobs, _p1_forward_kwargs = self._generate(
            pass1_prompts, images
        )

        # Slice prefix (tokens + logprobs) and decode to text for Pass 2
        prefix_ids_list: list[list[int]] = []
        prefix_lp_list: list[list[float]] = []
        continuation_prefixes: list[str] = []
        for i, ids in enumerate(p1_completion_ids):
            n = max(1, min(int(round(k * len(ids))), len(ids)))
            prefix_ids_list.append(list(ids[:n]))
            if p1_logprobs is not None:
                prefix_lp_list.append(list(p1_logprobs[i][:n]))
            # Decode without skipping specials so <think>/</think> round-trip into Pass 2's prompt
            continuation_prefixes.append(
                self.processing_class.decode(ids[:n], skip_special_tokens=False)
            )

        if self.accelerator.is_main_process:
            sample_p1 = self.processing_class.decode(p1_completion_ids[0], skip_special_tokens=False)
            print(f"[DEBUG SPLICE] starter={starter} k={k} pass1_len={len(p1_completion_ids[0])} prefix_len={len(prefix_ids_list[0])}")
            print(f"[DEBUG SPLICE] pass1[:300]={sample_p1[:300]}")
            print(f"[DEBUG SPLICE] prefix_text[:300]={continuation_prefixes[0][:300]}")

        # Pass 2: other role continues from the prefix
        p2_prompt_ids, p2_completion_ids, total_completion_tokens, p2_logprobs, forward_kwargs = self._generate(
            pass2_prompts, images, continuation_prefixes=continuation_prefixes
        )

        # Stitch completions: completion = prefix + pass2_suffix, truncated to max_completion_length
        max_len = self.max_completion_length
        stitched_completion_ids: list[list[int]] = []
        stitched_logprobs: Optional[list[list[float]]] = [] if p1_logprobs is not None and p2_logprobs is not None else None
        for i, (prefix_ids, suffix_ids) in enumerate(zip(prefix_ids_list, p2_completion_ids)):
            full_ids = (list(prefix_ids) + list(suffix_ids))[:max_len]
            stitched_completion_ids.append(full_ids)
            if stitched_logprobs is not None:
                full_lp = (list(prefix_lp_list[i]) + list(p2_logprobs[i]))[:max_len]
                assert len(full_ids) == len(full_lp), (
                    f"stitched id/logprob length mismatch at i={i}: ids={len(full_ids)} lp={len(full_lp)}"
                )
                stitched_logprobs.append(full_lp)

        # Programmatic verification: stitched[0] must equal prefix_ids_list[0] + p2_completion_ids[0] (truncated).
        # Crashes loudly if the stitch logic ever drifts out of sync with the returned completions.
        _ver_prefix_len = len(prefix_ids_list[0])
        _ver_stitched = stitched_completion_ids[0]
        _ver_expected_prefix = list(prefix_ids_list[0])
        _ver_expected_suffix = list(p2_completion_ids[0])[: max_len - _ver_prefix_len]
        assert list(_ver_stitched[:_ver_prefix_len]) == _ver_expected_prefix, (
            f"stitched prefix != prefix_ids_list: stitched[0][:{_ver_prefix_len}] differs from pass-1 prefix"
        )
        assert list(_ver_stitched[_ver_prefix_len:]) == _ver_expected_suffix, (
            f"stitched suffix != p2 suffix: stitched[0][{_ver_prefix_len}:] differs from pass-2 completion"
        )

        if self.accelerator.is_main_process:
            # Verify the suffix is actually from Pass 2 (not Pass 1's natural continuation).
            # Print three strings for sample 0 at the splice boundary. If splice is working,
            # `pass2_suffix` and `stitched_at_boundary` should be identical, and `pass1_natural`
            # should be visibly different (same prefix but different prompt conditioning).
            prefix_len = len(prefix_ids_list[0])
            pass1_natural_ids = list(p1_completion_ids[0])[prefix_len:prefix_len + 300]
            pass2_suffix_ids = list(p2_completion_ids[0])[:300]
            stitched_boundary_ids = list(stitched_completion_ids[0])[prefix_len:prefix_len + 300]
            pass1_natural_txt = self.processing_class.decode(pass1_natural_ids, skip_special_tokens=False) if pass1_natural_ids else "<empty: pass1 truncated exactly at prefix>"
            pass2_suffix_txt = self.processing_class.decode(pass2_suffix_ids, skip_special_tokens=False) if pass2_suffix_ids else "<empty: pass2 produced no tokens>"
            stitched_boundary_txt = self.processing_class.decode(stitched_boundary_ids, skip_special_tokens=False) if stitched_boundary_ids else "<empty>"
            print(f"[DEBUG SPLICE] --- SUFFIX PROVENANCE CHECK (sample 0) ---")
            print(f"[DEBUG SPLICE] pass1_natural_continuation[:300]={pass1_natural_txt}")
            print(f"[DEBUG SPLICE] pass2_actual_suffix[:300]={pass2_suffix_txt}")
            print(f"[DEBUG SPLICE] stitched_at_boundary[:300]={stitched_boundary_txt}")
            print(f"[DEBUG SPLICE] stitched==pass2? {stitched_boundary_ids == pass2_suffix_ids}")
            print(f"[DEBUG SPLICE] stitched==pass1_natural? {stitched_boundary_ids == pass1_natural_ids}")

        return p2_prompt_ids, stitched_completion_ids, total_completion_tokens, stitched_logprobs, forward_kwargs

    def _generate_critiques(self, critique_messages: list[list[dict]]) -> list[str]:
        """Generate critique text for a batch of prompts.

        Args:
            critique_messages: List of chat-formatted messages, each like
                [{"role": "user", "content": "..."}].

        Returns:
            List of critique text strings (thinking tags stripped).
        """
        chat_template_kwargs = {}
        if self.args.enable_thinking != "default":
            chat_template_kwargs["enable_thinking"] = (self.args.enable_thinking == "true")

        prompts_text = [
            maybe_apply_chat_template(
                {"prompt": msg, "chat_template_kwargs": chat_template_kwargs}, self.processing_class
            )["prompt"]
            for msg in critique_messages
        ]
        if self.args.force_thinking_prefix:
            prompts_text = [p + self.args.force_thinking_prefix for p in prompts_text]

        local_count = len(prompts_text)

        if self.use_vllm:
            if self.vllm_mode == "colocate" and self.args.vllm_enable_sleep_mode:
                torch.cuda.empty_cache()
                self.llm.wake_up()

            if self.vllm_mode == "server":
                all_prompts_text = gather_object(prompts_text)

                if self.accelerator.is_main_process:
                    with profiling_context(self, "vLLM.generate_critique"):
                        output = self.vllm_client.generate(
                            prompts=all_prompts_text,
                            n=1,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                            truncate_prompt_tokens=self.args.critique_max_prompt_length,
                            generation_kwargs=self.args.generation_kwargs,
                        )
                    payload = output["completion_ids"]
                else:
                    payload = None

                obj_list = [payload]
                broadcast_object_list(obj_list, from_process=0)
                all_completion_ids = obj_list[0]

                process_slice = slice(
                    self.accelerator.process_index * local_count,
                    (self.accelerator.process_index + 1) * local_count,
                )
                completion_ids = all_completion_ids[process_slice]

            elif self.vllm_mode == "colocate":
                generation_kwargs = {
                    "n": 1,
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": -1 if self.top_k is None else self.top_k,
                    "min_p": 0.0 if self.min_p is None else self.min_p,
                    "max_tokens": self.max_completion_length,
                    "truncate_prompt_tokens": self.args.critique_max_prompt_length,
                }
                if self.args.generation_kwargs is not None:
                    generation_kwargs.update(self.args.generation_kwargs)
                sampling_params = SamplingParams(**generation_kwargs)

                if self.vllm_tensor_parallel_size > 1:
                    orig_size = len(prompts_text)
                    gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                    torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.tp_group)
                    all_prompts_text_local = [p for sublist in gathered_prompts for p in sublist]
                else:
                    all_prompts_text_local = prompts_text

                with profiling_context(self, "vLLM.generate_critique"):
                    all_outputs = self.llm.generate(
                        all_prompts_text_local, sampling_params=sampling_params, use_tqdm=False
                    )

                all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

                if self.vllm_tensor_parallel_size > 1:
                    local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                    tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                    completion_ids = all_completion_ids[tp_slice]
                else:
                    completion_ids = all_completion_ids

                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=1)

        else:
            critique_inputs = self.processing_class(
                text=prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.args.critique_max_prompt_length,
                truncation=True,
                add_special_tokens=False,
            )
            critique_inputs = super()._prepare_inputs(critique_inputs)

            critique_gen_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k if self.top_k is not None else 0,
                do_sample=True,
                pad_token_id=self.pad_token_id,
            )

            with (
                profiling_context(self, "transformers.generate_critique"),
                unwrap_model_for_generation(
                    self.model_wrapped, self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **critique_inputs, generation_config=critique_gen_config, disable_compile=True
                )

            prompt_length = critique_inputs["input_ids"].size(1)
            raw_completion_ids = prompt_completion_ids[:, prompt_length:]

            is_eos = raw_completion_ids == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=raw_completion_ids.device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            seq_indices = torch.arange(is_eos.size(1), device=raw_completion_ids.device).expand(is_eos.size(0), -1)
            completion_mask = (seq_indices <= eos_idx.unsqueeze(1)).int()
            completion_ids = [c[m].tolist() for c, m in zip(raw_completion_ids, completion_mask.bool())]

        critiques = []
        for ids in completion_ids:
            text = self.processing_class.decode(ids, skip_special_tokens=False)
            text = strip_thinking_tokens(text)
            for sp_token in self.processing_class.all_special_tokens:
                text = text.replace(sp_token, '')
            critiques.append(text.strip())

        return critiques

    def _gate_status(self, completion_ids_list, gold_answers, wrong_answers):
        """Wrong-only gate: True where a completion is (a) terminated (not truncated),
        (b) gradeable, (c) incorrect vs the gold answer, and (d) when
        gate_require_diff_answer, math-inequivalent to the in-context wrong answer."""
        from evaluation.utils import extract_answer_math
        from evaluation.grader import math_equal

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        live = []
        for ids, gold, wrong in zip(completion_ids_list, gold_answers, wrong_answers):
            ids = list(ids)
            if len(ids) == 0 or ids[-1] not in eos_and_pad:
                live.append(False)  # truncated -> no final answer -> ungradeable
                continue
            text = self.processing_class.decode(ids, skip_special_tokens=True)
            pred = extract_answer_math(text)
            if not pred:
                live.append(False)
                continue
            try:
                is_correct = math_equal(pred, str(gold), timeout=True)
            except Exception:
                is_correct = False
            if is_correct:
                live.append(False)
                continue
            if self.args.gate_require_diff_answer and wrong:
                try:
                    same_as_ctx = math_equal(pred, str(wrong), timeout=True)
                except Exception:
                    same_as_ctx = False
                if same_as_ctx:
                    live.append(False)
                    continue
            live.append(True)
        return live

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]
        teacher_prompts = [x["teacher_prompt"] for x in inputs]

        if "images" in inputs[0]:
            images = [example.get("images") for example in inputs]
        elif "image" in inputs[0]:
            images = [[example.get("image")] if example.get("image") is not None else None for example in inputs]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # Process student prompts (always used for student training, regardless of generation source)
        chat_template_kwargs = {}
        if self.args.enable_thinking != "default":
            chat_template_kwargs["enable_thinking"] = (self.args.enable_thinking == "true")

        prompts_text = [
            maybe_apply_chat_template({"prompt": prompt, "chat_template_kwargs": chat_template_kwargs}, self.processing_class)["prompt"] for prompt in prompts
        ]
        if self.args.force_thinking_prefix:
            prompts_text = [p + self.args.force_thinking_prefix for p in prompts_text]
        if self.use_vllm:
            self.processing_class.truncation_side = "left"
        student_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        student_inputs = super()._prepare_inputs(student_inputs)
        student_prompt_ids, student_prompt_mask = student_inputs["input_ids"], student_inputs["attention_mask"]
        prompt_ids_list = [p[m].tolist() for p, m in zip(student_prompt_ids, student_prompt_mask.bool())]

        # Track demonstrations for critique path (gold answers, potentially updated by on-policy demos)
        demonstrations = [x.get("gold_answer", "") for x in inputs]

        # Process teacher prompts upfront when NOT using on-policy demos or critique (original behavior).
        # When using on-policy demos or critique, teacher prompts are constructed after generation.
        use_onpolicy_path = self.args.use_onpolicy_demos and getattr(self, 'demo_reward_fn', None) is not None
        use_critique_path = self.args.use_critique and getattr(self, 'demo_reward_fn', None) is not None
        if not use_onpolicy_path and not use_critique_path:
            teacher_prompts_text = [
                maybe_apply_chat_template({"prompt": prompt, "chat_template_kwargs": chat_template_kwargs}, self.processing_class)["prompt"] for prompt in teacher_prompts
            ]
            if self.args.force_thinking_prefix:
                teacher_prompts_text = [p + self.args.force_thinking_prefix for p in teacher_prompts_text]
            teacher_inputs = self.processing_class(
                text=teacher_prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
            )
            teacher_inputs = super()._prepare_inputs(teacher_inputs)
            if self.use_vllm:
                self.processing_class.truncation_side = "right"
            teacher_prompt_ids, teacher_prompt_mask = teacher_inputs["input_ids"], teacher_inputs["attention_mask"]
            teacher_prompt_ids_list = [p[m].tolist() for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())]

        if self.speculative_generation:
            # Speculative KD: student proposes blocks, teacher verifies via top-K acceptance
            with profiling_context(self, "speculative_generate"):
                completion_ids_tensor, acceptance_rate = self._speculative_generate(
                    student_prompt_ids, student_prompt_mask,
                    teacher_prompt_ids, teacher_prompt_mask,
                )

            # Mask everything after the first EOS token and convert to list-of-lists
            is_eos = completion_ids_tensor == self.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            seq_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask_tensor = (seq_indices <= eos_idx.unsqueeze(1)).int()
            completion_ids_list = [c[m].tolist() for c, m in zip(completion_ids_tensor, completion_mask_tensor.bool())]

            # Log generation metrics (mirrors _generate)
            prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids_list], device=device)
            completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
            agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
            agg_completion_lengths = self.accelerator.gather(completion_lengths)
            total_prompt_tokens = agg_prompt_lengths.sum()
            total_completion_tokens = agg_completion_lengths.sum()
            num_items_in_batch = total_completion_tokens

            if mode == "train":
                self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
            self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
            self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
            self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
            self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            agg_is_truncated = self.accelerator.gather(is_truncated)
            self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
            term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
            if len(term_completion_lengths) == 0:
                term_completion_lengths = torch.zeros(1, device=device)
            self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
            self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
            self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

            self._metrics[mode]["speculative/acceptance_rate"].append(acceptance_rate)

            sampling_per_token_logps_list = None
            forward_kwargs = {}

        elif self.args.splice_generation:
            # Chimeric two-pass splice: see _generate_spliced. Pass 1 generates with the starter's prompt;
            # Pass 2 continues from the first splice_k fraction of tokens using the other role's prompt.
            # The stitched completion (prefix + Pass 2 suffix) is used for the loss over the full sequence.
            (
                _generation_prompt_ids_list,
                completion_ids_list,
                num_items_in_batch,
                sampling_per_token_logps_list,
                forward_kwargs,
            ) = self._generate_spliced(prompts, teacher_prompts, images)

        else:
            # Standard generation path (vLLM / transformers generate)
            generation_prompts = teacher_prompts if self.generate_from_teacher else prompts

            (
                _generation_prompt_ids_list,
                completion_ids_list,
                num_items_in_batch,
                sampling_per_token_logps_list,
                forward_kwargs,
            ) = self._generate(generation_prompts, images)

        # Wrong-rollout gate (contrastive OPSD): the loss is applied only to rollouts
        # whose final answer is wrong and, optionally, different (math-equivalence)
        # from the known wrong answer already in the teacher context. Rows failing
        # the gate are regenerated for a bounded number of full-batch rounds; every
        # rank runs the same rounds so collective calls stay uniform (early exit is
        # itself a gathered, rank-uniform decision). Rows still failing after all
        # rounds are zero-masked below and excluded from the loss normalization.
        gate_live_mask = None
        if self.args.gate_mode == "wrong_only" and mode == "train" and not self.args.splice_generation:
            gate_golds = [x.get("gate_gold_answer", "") for x in inputs]
            gate_wrongs = [x.get("wrong_answer", "") for x in inputs]
            gate_live = self._gate_status(completion_ids_list, gate_golds, gate_wrongs)
            initial_live = sum(gate_live)
            rounds_used = 0
            for _round in range(self.args.gate_max_regen_rounds):
                all_live_local = torch.tensor(all(gate_live), device=device, dtype=torch.bool)
                if self.accelerator.gather(all_live_local).all().item():
                    break
                rounds_used += 1
                (
                    _regen_prompt_ids_list,
                    regen_completion_ids_list,
                    _regen_num_items,
                    regen_logps_list,
                    _regen_forward_kwargs,
                ) = self._generate(prompts, images)
                regen_live = self._gate_status(regen_completion_ids_list, gate_golds, gate_wrongs)
                for i in range(len(completion_ids_list)):
                    if not gate_live[i] and regen_live[i]:
                        completion_ids_list[i] = regen_completion_ids_list[i]
                        if sampling_per_token_logps_list is not None and regen_logps_list is not None:
                            sampling_per_token_logps_list[i] = regen_logps_list[i]
                        gate_live[i] = True
            gate_live_mask = torch.tensor(gate_live, dtype=torch.bool, device=device)
            agg_live = self.accelerator.gather(gate_live_mask.float().mean())
            self._metrics[mode]["gate/live_fraction"].append(agg_live.mean().item())
            self._metrics[mode]["gate/initial_live_fraction"].append(
                self.accelerator.gather(
                    torch.tensor(initial_live / max(len(gate_live), 1), device=device)
                ).mean().item()
            )
            self._metrics[mode]["gate/regen_rounds"].append(float(rounds_used))

        # On-policy demo: score completions and conditionally swap teacher prompts
        if use_onpolicy_path:
            decoded_completions = [
                self.processing_class.decode(ids, skip_special_tokens=False)
                for ids in completion_ids_list
            ]

            G = self.num_generations
            num_unique_prompts = len(prompts) // G
            swap_count = 0
            demo_scores = []

            template = Template(
                self.teacher_prompt_template
                .replace("{prompt}", "$prompt")
                .replace("{gold_answer}", "$gold_answer")
            )

            for group_start in range(0, len(prompts), G):
                raw_prompt = prompts[group_start][0]["content"]

                best_demo_text = None
                best_score = -1.0
                for j in range(G):
                    idx = group_start + j
                    score = self.demo_reward_fn(raw_prompt, decoded_completions[idx])
                    if self.accelerator.is_main_process:  # [DEBUG ONPOL] print #2: per-completion scores
                        print(f"[DEBUG ONPOL]   group={group_start//G} j={j} score={score:.3f} completion={decoded_completions[idx][:120]}...")
                    if score >= self.args.onpolicy_demo_reward_threshold and score > best_score:
                        best_score = score
                        best_demo_text = decoded_completions[idx]

                if self.accelerator.is_main_process:  # [DEBUG ONPOL] print #2b: best selection
                    print(f"[DEBUG ONPOL]   group={group_start//G} prompt={raw_prompt[:80]}... best_score={best_score:.3f} swapped={'YES' if best_demo_text is not None else 'NO'}")

                if best_demo_text is not None:
                    demo_text = best_demo_text
                    if self.args.strip_thinking_from_demo:
                        demo_text = strip_thinking_tokens(demo_text)
                    for sp_token in self.processing_class.all_special_tokens:
                        demo_text = demo_text.replace(sp_token, '')
                    demo_text = demo_text.strip()

                    if self.accelerator.is_main_process:  # [DEBUG ONPOL] print #3: cleaned demo text
                        print(f"[DEBUG ONPOL]   cleaned demo_text={demo_text[:200]}...")
                    new_content = template.substitute(prompt=raw_prompt, gold_answer=demo_text)
                    for j in range(G):
                        teacher_prompts[group_start + j] = [{"role": "user", "content": new_content}]
                        demonstrations[group_start + j] = demo_text
                    swap_count += 1
                    demo_scores.append(best_score)

            if self.accelerator.is_main_process:  # [DEBUG ONPOL] print #4: swap summary
                print(f"[DEBUG ONPOL] SWAP SUMMARY: swap_count={swap_count}/{num_unique_prompts} swap_rate={swap_count / max(num_unique_prompts, 1):.3f}")
                if demo_scores:
                    print(f"[DEBUG ONPOL]   mean_reward={sum(demo_scores)/len(demo_scores):.3f} scores={demo_scores}")

            self._metrics[mode]["onpolicy_demo/swap_rate"].append(swap_count / max(num_unique_prompts, 1))
            if demo_scores:
                self._metrics[mode]["onpolicy_demo/mean_reward"].append(sum(demo_scores) / len(demo_scores))

            # Tokenize teacher prompts (skip if critique path will rebuild them)
            if not use_critique_path:
                teacher_prompts_text = [
                    maybe_apply_chat_template({"prompt": prompt, "chat_template_kwargs": chat_template_kwargs}, self.processing_class)["prompt"] for prompt in teacher_prompts
                ]
                if self.args.force_thinking_prefix:
                    teacher_prompts_text = [p + self.args.force_thinking_prefix for p in teacher_prompts_text]
                teacher_inputs = self.processing_class(
                    text=teacher_prompts_text,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    max_length=self.max_prompt_length,
                    truncation=True,
                    add_special_tokens=False,
                )
                teacher_inputs = super()._prepare_inputs(teacher_inputs)
                if self.use_vllm:
                    self.processing_class.truncation_side = "right"
                teacher_prompt_ids, teacher_prompt_mask = teacher_inputs["input_ids"], teacher_inputs["attention_mask"]
                teacher_prompt_ids_list = [p[m].tolist() for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())]

        # Critique-conditioned self-teaching: generate critiques and rebuild teacher prompts
        if use_critique_path:
            decoded_completions_for_critique = [
                self.processing_class.decode(ids, skip_special_tokens=False)
                for ids in completion_ids_list
            ]

            raw_prompts = [p[0]["content"] for p in prompts]

            # Score each completion to determine correctness
            scores = []
            for i in range(len(prompts)):
                score = self.demo_reward_fn(raw_prompts[i], decoded_completions_for_critique[i])
                scores.append(score)
            is_correct = [s >= self.args.onpolicy_demo_reward_threshold for s in scores]

            all_correct = all(is_correct)
            num_correct = sum(is_correct)

            if self.accelerator.is_main_process:
                print(f"[DEBUG CRITIQUE] correct={num_correct}/{len(is_correct)} all_correct={all_correct}")

            # Build critique prompts for ALL rollouts (uniform batch size for distributed)
            critique_messages = []
            for i in range(len(prompts)):
                critique_content = self.args.critique_prompt_template.format(
                    problem_text=raw_prompts[i],
                    gold_solution_output=demonstrations[i],
                    student_thinking=decoded_completions_for_critique[i],
                )
                critique_messages.append([{"role": "user", "content": critique_content}])

            if all_correct:
                critiques = [self.args.critique_default_correct] * len(prompts)
            else:
                with profiling_context(self, "generate_critiques"):
                    critiques = self._generate_critiques(critique_messages)
                # Override correct rollouts with canned string
                for i in range(len(prompts)):
                    if is_correct[i]:
                        critiques[i] = self.args.critique_default_correct

            # Build teacher prompts using teacher_critique_template
            for i in range(len(prompts)):
                new_content = self.args.teacher_critique_template.format(
                    problem=raw_prompts[i],
                    gold_solution_output=demonstrations[i],
                    critique=critiques[i],
                )
                teacher_prompts[i] = [{"role": "user", "content": new_content}]

            if self.accelerator.is_main_process:
                print(f"[DEBUG CRITIQUE] Sample teacher prompt: {teacher_prompts[0][0]['content'][:500]}")
                if not all_correct:
                    first_incorrect = next(i for i, c in enumerate(is_correct) if not c)
                    print(f"[DEBUG CRITIQUE] Sample critique (idx={first_incorrect}): {critiques[first_incorrect][:300]}")

            # Log critique metrics
            self._metrics[mode]["critique/skip_rate"].append(num_correct / max(len(prompts), 1))
            if not all_correct:
                critique_lengths = [len(c) for i, c in enumerate(critiques) if not is_correct[i]]
                self._metrics[mode]["critique/mean_length"].append(
                    sum(critique_lengths) / max(len(critique_lengths), 1)
                )

            # Tokenize teacher prompts
            teacher_prompts_text = [
                maybe_apply_chat_template({"prompt": prompt, "chat_template_kwargs": chat_template_kwargs}, self.processing_class)["prompt"] for prompt in teacher_prompts
            ]
            if self.args.force_thinking_prefix:
                teacher_prompts_text = [p + self.args.force_thinking_prefix for p in teacher_prompts_text]
            teacher_inputs = self.processing_class(
                text=teacher_prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
            )
            teacher_inputs = super()._prepare_inputs(teacher_inputs)
            if self.use_vllm:
                self.processing_class.truncation_side = "right"
            teacher_prompt_ids, teacher_prompt_mask = teacher_inputs["input_ids"], teacher_inputs["attention_mask"]
            teacher_prompt_ids_list = [p[m].tolist() for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())]

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        teacher_prompt_ids = [torch.tensor(ids, device=device) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Zero out rollouts that failed the wrong-only gate after all regen rounds.
        if gate_live_mask is not None:
            completion_mask = completion_mask * gate_live_mask.unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        teacher_prompt_completion_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)  # (B, P+C)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)  # (B, P+C)
        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            # Skip when generate_from_teacher=True since importance sampling is not used in that case.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if not self.generate_from_teacher and (
                self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction)):
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    compute_all_logps=False,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            # Skip when generate_from_teacher=True since vLLM has teacher weights (no mismatch to correct)
            if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
            else:
                importance_sampling_ratio = None

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        compute_all_logps=False,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            compute_all_logps=False,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )   
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        # Not really necessary, but keeping for now
        rewards = torch.zeros_like(completion_ids, dtype=torch.float32)
        advantages = rewards
        
        # Keep a copy for logging (data is already local to each process, no slicing needed)
        all_process_advantages = advantages.clone()

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["rewards"]["main"].extend(gather_object(rewards.mean(dim=-1).tolist()))
        self._logs["advantages"].extend(gather_object(all_process_advantages.mean(dim=-1).tolist()))
        reward_to_log = rewards.clone()
        reward_to_log = reward_to_log[completion_mask.bool()]
        mean_reward = torch.mean(reward_to_log) if reward_to_log.numel() > 0 else torch.tensor(0.0, device=device)
        self._metrics[mode]["rewards"].append(self.accelerator.gather(mean_reward).mean().item())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if importance_sampling_ratio is not None:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]
        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output


    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The DistilTrainer does not support returning outputs")
        return self._compute_loss(model, inputs)

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        teacher_prompt_ids, teacher_prompt_mask = inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"]
        
        # Create a separate mask for loss computation that skips the first N tokens
        # Note: completion_mask is used for both attention (forward pass) and loss computation
        # We need to keep the original for attention, but create a modified one for loss
        loss_completion_mask = completion_mask
        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            # Create a mask that is 0 for the first num_loss_tokens_to_skip tokens and 1 elsewhere
            token_positions = torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            # Apply the skip mask (only mask tokens that were originally unmasked)
            loss_completion_mask = completion_mask * skip_mask
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, all_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        with torch.no_grad():
            teacher_per_token_logps, teacher_all_logps, teacher_entropies = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids,
                teacher_attention_mask,
                logits_to_keep,
                compute_entropy=True,
                temperature=self.teacher_temperature,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, loss_completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
        
        # Compute KL divergences using F.kl_div
        # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
        if self.alpha == 0: #Forward KL
            kl_loss = kl_div(all_logps, teacher_all_logps, reduction="none", log_target=True)
        elif self.alpha == 1: #Reverse KL
            kl_loss = kl_div(teacher_all_logps, all_logps, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            alpha = torch.tensor(self.alpha, dtype=all_logps.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack([all_logps + torch.log(1 - alpha), teacher_all_logps + torch.log(alpha)]),
                dim=0,
            )

            kl_teacher = kl_div(mixture_log_probs, teacher_all_logps, reduction="none", log_target=True)
            kl_student = kl_div(mixture_log_probs, all_logps, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            kl_loss = alpha * kl_teacher + (1 - alpha) * kl_student
        per_token_loss = kl_loss.sum(-1)

        if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
            ratio = inputs["importance_sampling_ratio"]
            importance_weights = (ratio * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
            importance_weights = importance_weights.unsqueeze(-1)
            per_token_loss = per_token_loss * importance_weights

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        per_seq_loss = (per_token_loss * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
        if self.args.gate_mode == "wrong_only":
            # Average over live (non-zero-mask) rows only, so gated-out rows do not
            # dilute the step: each step is a mean over the same effective batch of
            # gated rollouts regardless of how many rows the gate killed.
            live_rows = (loss_completion_mask.sum(-1) > 0).float()
            loss = (per_seq_loss * live_rows).sum() / live_rows.sum().clamp(min=1.0)
        else:
            loss = per_seq_loss.mean()
        loss = loss / self.current_gradient_accumulation_steps

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        with torch.no_grad():
            kl_approx = (per_token_logps - teacher_per_token_logps) + torch.exp(teacher_per_token_logps - per_token_logps) - 1
            kl_approx_mean = (kl_approx * loss_completion_mask).sum() / loss_completion_mask.sum()
        self._metrics[mode]["kl_approx"].append(self.accelerator.gather(kl_approx_mean).nanmean().item())
        
        loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * loss_completion_mask).sum() / loss_completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl_to_base_model"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                )

            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)] * len(self._logs["prompt"]),
                    "prompt": self._logs["prompt"],
                    "completion": self._logs["completion"],
                    **self._logs["rewards"],
                    "advantage": self._logs["advantages"],
                }

                if self._logs["images"]:
                    table["images"] = []
                    for image_list in self._logs["images"]:
                        # Convert images to wandb Image objects for proper visualization
                        table["images"].append([wandb.Image(image) for image in image_list])

                df = pd.DataFrame(table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])
                wandb.log({"completions": wandb.Table(dataframe=df)})

    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)


def prepare_distil_dataset(
    dataset: Dataset,
    prompt_key: str = "prompt",
    gold_answer_key: str = "answer",
    teacher_prompt_template: str = "{prompt}\n\nThis is an example for a response to the question:\n{gold_answer}\n\nNow answer with a response of your own, including the thinking process.",
    seed: int = 42,
) -> Dataset:
    """
    Prepare dataset for self-distillation training.
    
    Creates 'prompt' (student input) and 'teacher_prompt' (with demonstration) columns.
    
    Args:
        dataset: Input dataset with prompts and gold answers
        prompt_key: Key for the prompt column
        gold_answer_key: Key for the gold answer column  
        teacher_prompt_template: Template for teacher prompt (use {prompt} and {gold_answer})
        seed: Random seed for shuffling
        
    Returns:
        Dataset with 'prompt' and 'teacher_prompt' columns formatted for chat
    """
    def format_example(example):
        prompt = example.get(prompt_key, "")
        gold_answer = example.get(gold_answer_key, "")
        
        # Handle list-type gold answers (e.g., multiple steps)
        if isinstance(gold_answer, list):
            gold_answer = '\n'.join(str(item) for item in gold_answer)
        
        teacher_prompt_content = teacher_prompt_template.format(
            prompt=prompt,
            gold_answer=gold_answer
        )
        
        return {
            "prompt": [{"role": "user", "content": prompt}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt_content}],
        }
    
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    return dataset
