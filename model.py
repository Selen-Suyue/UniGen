import torch
import torch.nn as nn
import os
from transformers import ViTModel,BertModel,  BertConfig,  BertLMHeadModel
from dit import DiT
from torch.nn import functional as F    
from PIL import Image

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

        self.v_feat_proj= nn.Linear(config_text.vocab_size+2, d_model*4) 
        self.text_embeddings = nn.Embedding(config_text.vocab_size, d_model, padding_idx=config_text.pad_token_id)
        self.text_encoder = BertModel.from_pretrained(text_model_name, config=config_text)
        self.text_decoder = BertLMHeadModel.from_pretrained(text_model_name, config=config_text)

        vit_hidden_size = self.vit.config.hidden_size
        if vit_hidden_size != d_model:
             self.vit_proj = nn.Linear(vit_hidden_size, d_model)
        else:
             self.vit_proj = nn.Identity()
        self.upconv=nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)


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
    
    def forward_t2i(self, pixel_values, input_ids, attention_mask, labels):
        text_feat= self.encode_text_for_dit(input_ids, attention_mask) #16,768
        text_feat = text_feat.unsqueeze(1) 
        outputs = self.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=text_feat,
          ).logits #batch_size, seq_len, vocab_size
        v_feat = self.v_feat_proj(outputs)
        img = v_feat.reshape(v_feat.shape[0], 3, 256, 256) 
        img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img = self.upconv(img)
        loss = F.mse_loss(img, pixel_values)
        return loss,img
        

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
    
    def generate_image(self,name, input_ids, attention_mask):
        with torch.no_grad():
            text_feat = self.encode_text_for_dit(input_ids, attention_mask) 
            text_feat = text_feat.unsqueeze(1) 
            outputs = self.text_decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=text_feat,
            ).logits #batch_size, seq_len, vocab_size
            v_feat = self.v_feat_proj(outputs)
            img = v_feat.reshape(v_feat.shape[0], 3, 256, 256) 
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            generated_image = self.upconv(img)
            output_dir = 'gen_imgs'
            file_prefix = name
            for ig in range(generated_image.shape[0]):
                img_tensor = generated_image[ig].cpu()

                img_tensor = ((img_tensor.clamp(-1, 1) + 1) / 2) * 255
                img_np = img_tensor.permute(1, 2, 0).byte().numpy() 

                try:
                    img_pil = Image.fromarray(img_np, 'RGB')
                    filename = os.path.join(output_dir, f"{file_prefix}{ig+1}.png")
                    img_pil.save(filename)
                    print(f"Saved generated image {ig+1} to {filename}")
                except Exception as e:
                    print(f"Error saving image {ig+1}: {e}")

    

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None, mode='i2t'):
        if mode == 'i2t':
            if pixel_values is None or input_ids is None or attention_mask is None or labels is None:
                raise ValueError("Missing inputs for I2T mode")
            return self.forward_i2t(pixel_values, input_ids, attention_mask, labels)
        elif mode == 't2i':
            if input_ids is None or attention_mask is None:
                 raise ValueError("Missing inputs for T2I feature extraction mode")
            return self.forward_t2i(pixel_values, input_ids, attention_mask, labels)
        else:
             raise ValueError(f"Unknown mode: {mode}")
        
