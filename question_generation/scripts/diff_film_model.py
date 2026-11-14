"""
T5 with FiLM-style structural difficulty conditioning: a per-decoder-layer
scale (gamma) and shift (beta), derived from the difficulty tag, applied to
every decoder layer's hidden state -- not just a single additive bias before
the LM head (diff_cond_model.py's approach, which only weakly broke beam
search's collapse to one tag-invariant output on about half of test
passages, even with a 20x learning-rate boost on the bias vector). FiLM
modulates the computation at every layer, giving the signal many more
chances to actually steer what gets generated.

Implemented via forward hooks on each decoder T5Block (T5Stack runs blocks
sequentially, feeding each block's output into the next; a hook that returns
a replacement tuple substitutes what the next block sees) rather than
reimplementing T5Stack's internals (relative position bias, layer norm,
KV-cache bookkeeping) by hand.

gamma/beta are initialized to produce the identity transform (gamma=1, beta=0)
so a freshly loaded/untrained model behaves exactly like the plain checkpoint
until training actually moves these parameters.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

NUM_DIFFICULTY_CLASSES = 3
DIFF2ID = {"EASY": 0, "MEDIUM": 1, "HARD": 2}


class T5WithDifficultyFiLM(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.num_decoder_layers = config.num_decoder_layers
        self.film_d_model = config.d_model
        self.difficulty_film = nn.Embedding(
            NUM_DIFFICULTY_CLASSES, self.num_decoder_layers * 2 * self.film_d_model
        )
        nn.init.zeros_(self.difficulty_film.weight)

        self._film_gamma: Optional[torch.Tensor] = None
        self._film_beta: Optional[torch.Tensor] = None
        for layer_idx, block in enumerate(self.decoder.block):
            block.register_forward_hook(self._make_film_hook(layer_idx))

    def _make_film_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if self._film_gamma is None:
                return None
            gamma = self._film_gamma[:, layer_idx, :].unsqueeze(1)  # (batch, 1, d_model)
            beta = self._film_beta[:, layer_idx, :].unsqueeze(1)
            modulated = output[0] * gamma + beta
            return (modulated,) + tuple(output[1:])
        return hook

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # Only force the identity-transform zero-init when difficulty_film is
        # genuinely absent from the checkpoint (loading the plain base model).
        # Previously this zeroed the weight unconditionally, which silently
        # discarded already-trained difficulty_film weights every time the
        # model's OWN finetuned checkpoint was loaded -- the actual cause of
        # the EASY/MEDIUM/HARD beam-search collapse seen at inference.
        pretrained_model_name_or_path = args[0] if args else kwargs.get("pretrained_model_name_or_path")
        from pathlib import Path
        import json as _json
        has_film_weights = False
        if pretrained_model_name_or_path is not None:
            safetensors_index = Path(str(pretrained_model_name_or_path)) / "model.safetensors.index.json"
            safetensors_file = Path(str(pretrained_model_name_or_path)) / "model.safetensors"
            if safetensors_index.exists():
                has_film_weights = "difficulty_film.weight" in _json.loads(safetensors_index.read_text()).get("weight_map", {})
            elif safetensors_file.exists():
                from safetensors import safe_open
                with safe_open(str(safetensors_file), framework="pt") as f:
                    has_film_weights = "difficulty_film.weight" in f.keys()
        model = super().from_pretrained(*args, **kwargs)
        if not has_film_weights:
            nn.init.zeros_(model.difficulty_film.weight)
        return model

    def _set_film(self, difficulty_ids: Optional[torch.LongTensor]) -> None:
        if difficulty_ids is None:
            self._film_gamma = None
            self._film_beta = None
            return
        raw = self.difficulty_film(difficulty_ids)  # (batch, num_layers*2*d_model)
        raw = raw.view(raw.shape[0], self.num_decoder_layers, 2, self.film_d_model)
        self._film_gamma = 1.0 + raw[:, :, 0, :]
        self._film_beta = raw[:, :, 1, :]

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

        self._set_film(difficulty_ids)
        try:
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
        finally:
            self._film_gamma = None
            self._film_beta = None

        sequence_output = decoder_outputs[0]

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
