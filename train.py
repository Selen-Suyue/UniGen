import torch
import torch.optim as optim
import random
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from accelerate import Accelerator 
from dataset import Flickr8kDataset, collate_fn
from model import MultimodalTransformer
from termcolor import cprint

IMAGE_DIR = 'flickr8k/Images'
CAPTIONS_FILE = 'flickr8k/captions.txt'
VIT_MODEL = 'google/vit-base-patch16-224-in21k'
TEXT_MODEL = 'bert-base-uncased' 
BATCH_SIZE = 16 
LEARNING_RATE = 5e-5
EPOCHS = 10
MAX_LENGTH = 64 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.9 
ACCELERATOR_LOGGING_DIR = "logs" 
SAVE_CHECKPOINT_PATH = "UniGen.pt"

def main():
    print(f"Using device: {DEVICE}")

    accelerator = Accelerator(log_with="tensorboard", project_dir=ACCELERATOR_LOGGING_DIR)
    accelerator.print(f"Accelerator device: {accelerator.device}")


    tokenizer = BertTokenizerFast.from_pretrained(TEXT_MODEL)
   
    if tokenizer.bos_token is None: tokenizer.add_special_tokens({'bos_token': '[SOS]'})
    if tokenizer.eos_token is None: tokenizer.add_special_tokens({'eos_token': '[EOS]'})


    model = MultimodalTransformer(
        vit_model_name=VIT_MODEL,
        text_model_name=TEXT_MODEL,
        d_model=768, 
    )
    
    model.text_decoder.resize_token_embeddings(len(tokenizer))
    model.text_encoder.resize_token_embeddings(len(tokenizer)) 


    train_dataset = Flickr8kDataset(IMAGE_DIR, CAPTIONS_FILE, tokenizer, max_length=MAX_LENGTH, split='train', train_ratio=TRAIN_RATIO)
    val_dataset = Flickr8kDataset(IMAGE_DIR, CAPTIONS_FILE, tokenizer, max_length=MAX_LENGTH, split='val', train_ratio=TRAIN_RATIO)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, 
                                  num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, 
                                num_workers=4, pin_memory=True)


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)


    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps), 
        num_training_steps=num_training_steps
    )


    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    accelerator.init_trackers("flickr8k_UniGen")


    global_step = 0
    for epoch in range(EPOCHS):
        model.train()
        total_i2t_loss = 0
        for step, batch in enumerate(train_dataloader):
             if batch is None: continue 
             optimizer.zero_grad()
             pixel_values = batch['pixel_values']
             input_ids = batch['input_ids']
             attention_mask = batch['attention_mask']

             labels = input_ids.clone()
             labels[labels == tokenizer.pad_token_id] = -100 
             i2t_loss, _ = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels, mode='i2t')
             loss = i2t_loss
             accelerator.backward(loss)
             optimizer.step()
             lr_scheduler.step()

             total_i2t_loss += loss.item()
             global_step += 1

             if accelerator.is_main_process and (step + 1) % 50 == 0:
                 avg_loss = total_i2t_loss / (step + 1)
                 accelerator.print(f"Epoch {epoch+1}/{EPOCHS}, Step {step+1}/{len(train_dataloader)}, I2T Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                 accelerator.log({"train_i2t_loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)
        ####VAL#####
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                 if batch is None: continue
                 pixel_values = batch['pixel_values']
                 input_ids = batch['input_ids']
                 attention_mask = batch['attention_mask']
                 labels = input_ids.clone()
                 labels[labels == tokenizer.pad_token_id] = -100


                 i2t_loss, _ = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, 
                                     labels=labels, mode='i2t')
                 total_val_loss += i2t_loss.item()


                 if accelerator.is_main_process and random.random() < 0.5:
                     try:
                          unwrapped_model = accelerator.unwrap_model(model)
                          generated_captions = unwrapped_model.generate_caption(pixel_values[:2], tokenizer, max_length=MAX_LENGTH)
                          accelerator.print("\n--- Example Generation ---")
                          for i in range(len(generated_captions)):
                                accelerator.print(f"Image ID: {batch['image_ids'][i]}")
                                accelerator.print(f"  Original: {batch['captions'][i]}")
                                accelerator.print(f"  Generated: {generated_captions[i]}")
                          accelerator.print("------------------------\n")
                     except Exception as e:
                          accelerator.print(f"Generation failed: {e}")


        avg_val_loss = total_val_loss / len(val_dataloader)
        accelerator.print(f"Epoch {epoch+1} Validation I2T Loss: {avg_val_loss:.4f}")
        if accelerator.is_main_process:
             accelerator.log({"val_i2t_loss": avg_val_loss}, step=global_step)

             accelerator.wait_for_everyone()
             unwrapped_model = accelerator.unwrap_model(model)
             accelerator.save({
                    "epoch": epoch,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "loss": avg_val_loss,
                }, SAVE_CHECKPOINT_PATH)
             accelerator.print(f"Checkpoint saved to {SAVE_CHECKPOINT_PATH}")

    accelerator.end_training()
    print("Training finished.")


if __name__ == "__main__":
    main()