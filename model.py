import torch
import torch.nn as nn
from transformers import ViTModel, BertConfig, BertModel, BertLMHeadModel, PreTrainedTokenizerFast
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

class MultimodalTransformer(nn.Module):
    def __init__(self, vit_model_name='google/vit-base-patch16-224-in21k', text_model_name='bert-base-uncased', d_model=768, 
                 nhead=12, num_decoder_layers=6):
        super().__init__()

        self.vit = ViTModel.from_pretrained(vit_model_name)

        config_text = BertConfig.from_pretrained(text_model_name)
        config_text.is_decoder = True
        config_text.add_cross_attention = True
        config_text.vocab_size = config_text.vocab_size 


        self.text_embeddings = nn.Embedding(config_text.vocab_size, d_model, padding_idx=config_text.pad_token_id)
        self.text_encoder = BertModel.from_pretrained(text_model_name, config=config_text) 
        self.text_decoder = BertLMHeadModel.from_pretrained(text_model_name, config=config_text)

        vit_hidden_size = self.vit.config.hidden_size
        if vit_hidden_size != d_model:
             self.vit_proj = nn.Linear(vit_hidden_size, d_model)
        else:
             self.vit_proj = nn.Identity()


    def encode_image(self, pixel_values):
        vit_outputs = self.vit(pixel_values=pixel_values)
        image_embeds = vit_outputs.last_hidden_state
        image_embeds_projected = self.vit_proj(image_embeds)
        image_attention_mask = torch.ones(image_embeds_projected.size()[:-1], dtype=torch.long, device=pixel_values.device)
        return image_embeds_projected, image_attention_mask

    def encode_text_for_dit(self, input_ids, attention_mask):
        encoder_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_features = encoder_outputs.pooler_output 
        return text_features 

    def forward_i2t(self, pixel_values, input_ids, attention_mask, labels):
        encoder_hidden_states, encoder_attention_mask = self.encode_image(pixel_values)

        decoder_outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels
        )
        return decoder_outputs.loss, decoder_outputs.logits

    def generate_caption(self, pixel_values, tokenizer, max_length=50, num_beams=4):
         encoder_hidden_states, encoder_attention_mask = self.encode_image(pixel_values)
         batch_size = pixel_values.shape[0]

         decoder_start_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
         decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=pixel_values.device)
         attention_mask = torch.ones_like(decoder_input_ids)


         outputs = self.text_decoder.generate(
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.sep_token_id, 
                bos_token_id=decoder_start_token_id,
                early_stopping=True,
          )

         return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, mode='i2t'):
        if mode == 'i2t':
            if pixel_values is None or input_ids is None or attention_mask is None or labels is None:
                raise ValueError("Missing inputs for I2T mode")
            return self.forward_i2t(pixel_values, input_ids, attention_mask, labels)
        elif mode == 't2i_features':
            if input_ids is None or attention_mask is None:
                 raise ValueError("Missing inputs for T2I feature extraction mode")
            return self.encode_text_for_dit(input_ids, attention_mask)
        elif mode == 'generate':
             if pixel_values is None:
                 raise ValueError("Missing pixel_values for generation mode")
             raise NotImplementedError("Call generate_caption method directly for generation")
        else:
             raise ValueError(f"Unknown mode: {mode}")