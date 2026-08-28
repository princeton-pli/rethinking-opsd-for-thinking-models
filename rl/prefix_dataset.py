# Custom verl dataset: feed a pre-rendered plain-string prompt VERBATIM.
#
# Needed because our episodes end mid-generation -- inside an open <think>
# block -- and the stock pipeline unconditionally applies the chat template
# with add_generation_prompt=True, which cannot express that (Qwen3's template
# force-closes assistant turns with </think>/<|im_end|> and appends a fresh
# assistant header).
#
# verl 0.7.0 (the della runtime) restructured this completely vs the
# verl@083da9ab this file was first written against:
#   * The SPMD sync rollout is RETIRED (vllm_rollout.py:generate_sequences
#     raises NotImplementedError, "SPMD mode was retired in PR #4411");
#     ray_trainer.py sets `self.async_rollout_mode = True` unconditionally
#     ("mode is always 'async' since sync mode is deprecated"). Do NOT pass
#     rollout.mode=sync -- there is no sync mode; generation always goes
#     through the AgentLoop (default agent: single_turn_agent).
#   * RLHFDataset.__getitem__ no longer tokenizes at all. It returns
#     `raw_prompt` (a messages list) plus `dummy_tensor` / `index` /
#     `tools_kwargs` / `interaction_kwargs`; `raw_prompt_ids` is consumed
#     NOWHERE in 0.7.0. Tokenization happens inside the rollout worker:
#     SingleTurnAgentLoop.run does
#         prompt_ids = await self.apply_chat_template(messages, ...)
#     where AgentLoopBase.apply_chat_template calls
#         tokenizer.apply_chat_template(messages, tools=tools,
#             add_generation_prompt=True, tokenize=True,
#             **self.apply_chat_template_kwargs)          # agent_loop.py:295-304
#     with apply_chat_template_kwargs read from `data.apply_chat_template_kwargs`.
#
# So in 0.7.0 the dataset alone CANNOT force verbatim tokenization; the
# template override must reach the AgentLoopWorker via config. This class
# handles the plain-string prompt column (wraps it into a 1-message list and
# filters on the string's true token length), and the launch config MUST add
# the identity chat template so the worker tokenizes the string verbatim
# (apply_chat_template tokenizes its rendered output with
# add_special_tokens=False, so ids == tokenizer.encode(prompt,
# add_special_tokens=False) and the policy continues exactly from our prefix):
#
#   data.custom_cls.path=<abs path to this file>
#   data.custom_cls.name=PrefixContinuationDataset
#   data.prompt_key=prompt          # a plain-string column
#   +data.apply_chat_template_kwargs.chat_template=<IDENTITY_CHAT_TEMPLATE below>
#     (hydra CLI quoting of the Jinja braces is fiddly; safest is a config
#      yaml or: TPL="{% for m in messages %}{{ m['content'] }}{% endfor %}"
#      then "+data.apply_chat_template_kwargs.chat_template=$TPL" -- verify
#      the resolved config echoes it back exactly before a real run)
#
# (i.e. the Jinja template IDENTITY_CHAT_TEMPLATE below; it ignores
# add_generation_prompt/tools and emits message content verbatim. Equivalent
# stock alternative: actor_rollout_ref.model.custom_chat_template=<same>,
# which AgentLoopWorker.__init__ assigns to its tokenizer. If the parquet
# were rewritten to a messages-format prompt column, the stock RLHFDataset +
# this config kwarg would suffice and this subclass would be unnecessary;
# it survives only to adapt the existing plain-string parquet.)
# See scratchpad memo / SPEC_grader_rl.md §5.
import torch

from verl.utils.dataset.rl_dataset import RLHFDataset

# Emits the concatenated message contents verbatim; must be passed as
# data.apply_chat_template_kwargs.chat_template in the launch config.
IDENTITY_CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


class PrefixContinuationDataset(RLHFDataset):
    def maybe_filter_out_long_prompts(self, dataframe=None):
        # Stock doc2len calls tokenizer.apply_chat_template(doc[prompt_key], ...)
        # which requires a messages list and would throw on our plain string
        # (the except branch then silently filters out EVERY row). Measure the
        # string's true token length instead -- identical to what the
        # AgentLoop produces under IDENTITY_CHAT_TEMPLATE.
        if self.filter_overlong_prompts:
            tok, key, maxlen = self.tokenizer, self.prompt_key, self.max_prompt_length
            dataframe = dataframe.filter(
                lambda doc: len(tok.encode(doc[key], add_special_tokens=False)) <= maxlen,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {maxlen} tokens (plain string)",
            )
            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def __getitem__(self, item):
        # Mirrors RLHFDataset.__getitem__ (verl 0.7.0), except raw_prompt wraps
        # our pre-rendered plain string as a single message so that
        # SingleTurnAgentLoop's `list(kwargs["raw_prompt"])` yields
        # [{role, content}] (a bare string would be exploded into characters).
        # No input_ids/attention_mask here: 0.7.0's trainer keeps dataset
        # tensors in the batch and unions them with the rollout output, so
        # emitting input_ids would collide with the AgentLoop's own.
        row_dict: dict = dict(self.dataframe[item])
        prompt_str: str = row_dict[self.prompt_key]
        row_dict["raw_prompt"] = [{"role": "user", "content": prompt_str}]

        # Dummy tensor keeps DataProto.batch non-empty (as in stock 0.7.0).
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        row_dict["index"] = row_dict["extra_info"].get("index", 0)
        row_dict["tools_kwargs"] = row_dict["extra_info"].get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = row_dict["extra_info"].get("interaction_kwargs", {})
        return row_dict
