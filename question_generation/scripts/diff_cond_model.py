"""
T5 with STRUCTURAL difficulty conditioning.

Earlier experiments (sample_irt_10k_generations.py, with and without
classifier-free-guidance-style decode-time amplification) showed that a
<EASY>/<MEDIUM>/<HARD> tag prepended to the input text has no reliable effect
on beam-search output: the tag is one token competing with ~1024 tokens of
passage inside the encoder, and cross-attention lets the decoder effectively
ignore it. Decode-time tricks (CFG guidance) only produced degenerate output,
not real differentiation -- confirming the signal is too weak at the model
level, not just masked by greedy decoding.

This module instead injects a learned difficulty embedding directly into the
DECODER's hidden state, at every position, right before the LM head -- so the
signal cannot be out-competed by the passage during cross-attention. The
<EASY>/<MEDIUM>/<HARD> input tag is left in place too (redundant, harmless).

Forward signature mirrors transformers' T5ForConditionalGeneration.forward
(same version this repo pins, 4.57.6 / 5.8.0) with one addition:
`difficulty_ids` (long tensor of shape (batch,), values in {0,1,2}).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

NUM_DIFFICULTY_CLASSES = 3
DIFF2ID = {"EASY": 0, "MEDIUM": 1, "HARD": 2}


class T5WithDifficultyConditioning(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.difficulty_embedding = nn.Embedding(NUM_DIFFICULTY_CLASSES, config.d_model)
        # Small init so early training doesn't get destabilized by a large additive bias
        nn.init.normal_(self.difficulty_embedding.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # transformers' "missing key" handling for a newly-added param zeroes
        # out difficulty_embedding.weight after construction (it has no
        # recognized _init_weights branch), silently undoing the std=0.02
        # init above -- reapply it explicitly every load, from_pretrained is
        # the only entry point that matters here.
        model = super().from_pretrained(*args, **kwargs)
        nn.init.normal_(model.difficulty_embedding.weight, mean=0.0, std=0.02)
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        difficulty_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        # The structural fix: add the difficulty signal directly to every
        # decoder position, right before the LM head -- bypassing the
        # encoder/cross-attention path where it previously got diluted by
        # the much longer passage.
        if difficulty_ids is not None:
            bias = self.difficulty_embedding(difficulty_ids)  # (batch, d_model)
            sequence_output = sequence_output + bias.unsqueeze(1)

        # NOTE: don't trust self.config.tie_word_embeddings here -- for a
        # checkpoint whose lm_head/shared weights were saved untied (this
        # one), the base T5ForConditionalGeneration class gets a proper
        # fixup during loading that flips this flag; our subclass doesn't
        # reliably get the same fixup, so check actual tensor identity
        # instead -- this directly matches whether the base class would
        # apply this scale.
        actually_tied = self.lm_head.weight.data_ptr() == self.shared.weight.data_ptr()
        if actually_tied:
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
