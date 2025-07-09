import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Union, Optional
import logging

from transformers.modeling_utils import PreTrainedModel
from .graph_t5 import T5ForConditionalGeneration
from .graph_t5 import T5Config, T5EncoderModel
from .graph_t5 import T5TokenizerFast as T5Tokenizer
from .wrapper_functions import graph_to_graphT5, get_dummy_graph

class GraphT5ForConditionalGeneration(PreTrainedModel):
    config_class = T5Config

    def __init__(
        self,
        config: T5Config,
    ):
        super().__init__(config=config)
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.modelsize, model_max_length=self.config.model_max_length
        )

        self.t5model = T5ForConditionalGeneration.from_pretrained(
            self.config.modelsize, config=config, ignore_mismatched_sizes=True
        )  # when initializing the model with .from_pretrained, the weights are loaded from the pretrained model, so the t5 parameters are not actually used in that case. Loading them here is unnecessary overhead.
        self.t5model.encoder = T5EncoderModel.from_pretrained(
            self.config.modelsize, config=config, ignore_mismatched_sizes=True
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
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        logits = self.t5model(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
            labels=labels,
        )  # (batch_size, seq_len, hidden_size)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        relative_position: torch.Tensor,
        sparsity_mask: torch.Tensor,
        use_additional_bucket: torch.Tensor,
        max_length: int = 512,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 1,
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

class GraphT5ForSequenceClassification(PreTrainedModel):
    config_class = T5Config

    def __init__(self, config: T5Config):
        super().__init__(config=config)
        self.config = config
        self.tokenizer = T5Tokenizer.from_pretrained(
            self.config.modelsize, model_max_length=self.config.model_max_length
        )
        self.t5model = T5EncoderModel.from_pretrained(
            self.config.modelsize, config=config, ignore_mismatched_sizes=True
        )
        self.hidden_size = self.t5model.config.d_model
        self.classification_head = nn.Linear(self.hidden_size, self.config.num_classes, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def get_config(num_classes: int, modelsize: str = "t5-base", num_additional_buckets: int = 0, model_max_length: int = 512) -> T5Config:
        config = T5Config.from_pretrained(modelsize)
        config.num_classes = int(num_classes)
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
        logging.debug('t5 encoder model')
        output = self.t5model(
            input_ids=input_ids,
            relative_position=relative_position,
            sparsity_mask=sparsity_mask,
            use_additional_bucket=use_additional_bucket,
        )  # output[0]: (batch_size, seq_len, hidden_size)

        # Sequence classification: use mean pooling
        sequence_output = output[0]  # (batch_size, seq_len, hidden_size)
        pooled_output = sequence_output.mean(dim=1)  # (batch_size, hidden_size)

        logging.debug('classification head')
        logits = self.classification_head(pooled_output)  # (batch_size, num_classes)
        return logits

    def get_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return self.softmax(logits)  # (batch_size, num_classes)

    def get_label(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1) 