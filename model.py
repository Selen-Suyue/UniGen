import torch
import torch.nn as nn
import os
from transformers import ViTModel,BertModel,  BertConfig,  BertLMHeadModel
from torch.nn import functional as F    
from PIL import Image
from diffusers import DDPMScheduler, UNet2DModel
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_CHECKPOINT_PATH = "ckpt/UniGen.pt"

class IntegratedImageGenerator(nn.Module):
    def __init__(self, unigen_model_path="ckpt/UniGen.pt", image_size=224, unet_channels=3, unet_model_channels=128):
        super().__init__()

        self.unigen = MultimodalTransformer()
        self.load_unigen_weights(unigen_model_path)
        self.freeze_unigen()

        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=unet_channels*2,
            out_channels=unet_channels,
            layers_per_block=2,
            block_out_channels=(unet_model_channels, unet_model_channels * 2, unet_model_channels * 4, unet_model_channels * 4),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.feature_projection = nn.Linear(768, unet_model_channels * 4)

    def load_unigen_weights(self, unigen_model_path):
        if os.path.exists(unigen_model_path):
            self.unigen.text_decoder.resize_token_embeddings(30524)
            self.unigen.text_encoder.resize_token_embeddings(30524)
            checkpoint = torch.load(SAVE_CHECKPOINT_PATH, map_location=DEVICE)
            if 'model_state_dict' in checkpoint:
                self.unigen.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.unigen.load_state_dict(checkpoint)

    def freeze_unigen(self):
        for param in self.unigen.parameters():
            param.requires_grad = False
        print("UniGen weights frozen.")

    def forward_unet(self, noisy_latents, timesteps, text_features):
        projected_text_features = text_features
        model_input = torch.cat([noisy_latents, projected_text_features], dim=1)
        noise_pred = self.unet(model_input, timesteps).sample
        return noise_pred

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, mode='train_image_generation'):
        if mode == 'train_image_generation':
            if pixel_values is None or input_ids is None or attention_mask is None:
                raise ValueError("Missing inputs for train_image_generation mode")

            with torch.no_grad():
                _, text_features = self.unigen.forward_t2i(input_ids=input_ids, attention_mask=attention_mask)

            noise = torch.randn(pixel_values.shape, device=pixel_values.device)
            bsz = pixel_values.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (bsz,), device=pixel_values.device).long()
            noisy_latents = self.noise_scheduler.add_noise(pixel_values, noise, timesteps)

            noise_pred = self.forward_unet(noisy_latents, timesteps, text_features)
            loss = F.mse_loss(noise_pred, noise)
            return loss

        elif mode == 'generate_image':
            if input_ids is None or attention_mask is None:
                 raise ValueError("Missing inputs for generate_image mode")

            with torch.no_grad():
                _, text_features = self.unigen.forward_t2i(input_ids=input_ids, attention_mask=attention_mask)

            latent = torch.randn((input_ids.shape[0], self.unet.config.out_channels, self.unet.config.sample_size, self.unet.config.sample_size), device=self.feature_projection.weight.device)

            for t in self.noise_scheduler.timesteps:
                noise_pred = self.forward_unet(latent, t, text_features)
                latent = self.noise_scheduler.step(noise_pred, t, latent).prev_sample

            generated_image = ((latent.clamp(-1, 1) + 1) / 2)

            return generated_image

        else:
             raise ValueError(f"Unknown mode: {mode}")
        
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
    
    def forward_t2i(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
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
        if pixel_values is not None:
            loss = F.mse_loss(img, pixel_values)
        else:
            loss = None
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
        
