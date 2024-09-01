import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union, Optional
import logging

# from transformers import T5ForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel

from models.graph_T5.graph_t5 import T5ForConditionalGeneration
from models.graph_T5.graph_t5 import T5Config, T5EncoderModel
from models.graph_T5.graph_t5 import T5TokenizerFast as T5Tokenizer
from models.graph_T5.wrapper_functions import graph_to_graphT5, get_dummy_graph


class GraphT5ForConditionalGeneration(PreTrainedModel):
    config_class = T5Config

    def __init__(self, config: T5Config, encoder_model_size: str = "t5-large"):
        super().__init__(config=config)
        self.config = config
        self.encoder_model_size = encoder_model_size
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.modelsize, model_max_length=self.config.model_max_length
        )
        if "flan" in self.config.modelsize:
            ignore_mismatched_sizes = False
        else:
            ignore_mismatched_sizes = True

        self.t5model = T5ForConditionalGeneration.from_pretrained(
            self.config.modelsize,
            config=config,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )  # when initializing the model with .from_pretrained, the weights are loaded from the pretrained model, so the t5 parameters are not actually used in that case. Loading them here is unnecessary overhead.
        self.t5model.encoder = T5EncoderModel.from_pretrained(
            self.encoder_model_size, config=config, ignore_mismatched_sizes=True
        )
        self.hidden_size = self.t5model.config.d_model
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_config(
        modelsize: str = "t5-base",
        num_additional_buckets: int = 0,
        model_max_length: int = 512,
    ) -> T5Config:
        config = T5Config.from_pretrained(modelsize)
        config.modelsize = str(modelsize)
        config.relative_attention_num_additional_buckets = int(num_additional_buckets)
        config.model_max_length = int(model_max_length)
        return config

    def forward(
        self,
        input_ids: torch.Tensor,
        relative_position: torch.Tensor,
        sparsity_mask: torch.Tensor,
        use_additional_bucket: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.t5model(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
        )  # (batch_size, seq_len, hidden_size)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        relative_position: torch.Tensor,
        sparsity_mask: torch.Tensor,
        use_additional_bucket: torch.Tensor,
        max_length: int = 128,
        num_beams: int = 4,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2,
        length_penalty: float = 1.0,
        use_cache: bool = True,
        pad_token_id: int = None,
        eos_token_id: int = None,
        **model_kwargs,
    ) -> torch.Tensor:
        return self.t5model.generate(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **model_kwargs,
        )
