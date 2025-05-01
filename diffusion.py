import torch
import torch.optim as optim
import random
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from accelerate import Accelerator
from dataset import Flickr8kDataset, collate_fn
from model import IntegratedImageGenerator
from termcolor import cprint
import os
from PIL import Image
from tqdm import tqdm
# accelerate launch --num_processes 1 --gpu_ids "6" --num_machines 1 --mixed_precision no --dynamo_backend no diffusion.py
# accelerate launch --num_processes 2 --multi_gpu --gpu_ids "6,7" --mixed_precision no --dynamo_backend no diffusion.py
IMAGE_DIR = 'flickr8k/Images'
CAPTIONS_FILE = 'flickr8k/captions.txt'
VIT_MODEL = 'google/vit-base-patch16-224-in21k'
TEXT_MODEL = 'bert-base-uncased'
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
EPOCHS = 100
MAX_LENGTH = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_RATIO = 0.95
ACCELERATOR_LOGGING_DIR = "logs"
SAVE_CHECKPOINT_PATH = "ckpt/IntegratedImageGenerator.pt"
UNIGEN_CHECKPOINT_PATH = "ckpt/UniGen.pt"
RESUME_TRAINING = True  # 设置为 True 表示从 checkpoint 恢复训练
RESUME_CHECKPOINT_PATH = "ckpt/IntegratedImageGenerator.pt"

def main():
    print(f"Using device: {DEVICE}")

    accelerator = Accelerator(log_with="tensorboard", project_dir=ACCELERATOR_LOGGING_DIR)
    accelerator.print(f"Accelerator device: {accelerator.device}")

    tokenizer = BertTokenizerFast.from_pretrained(TEXT_MODEL)

    if tokenizer.bos_token is None: tokenizer.add_special_tokens({'bos_token': '[SOS]'})
    if tokenizer.eos_token is None: tokenizer.add_special_tokens({'eos_token': '[EOS]'})

    model = IntegratedImageGenerator(unigen_model_path=UNIGEN_CHECKPOINT_PATH)

    train_dataset = Flickr8kDataset(IMAGE_DIR, CAPTIONS_FILE, tokenizer, max_length=MAX_LENGTH, split='train', train_ratio=TRAIN_RATIO)
    val_dataset = Flickr8kDataset(IMAGE_DIR, CAPTIONS_FILE, tokenizer, max_length=MAX_LENGTH, split='val', train_ratio=TRAIN_RATIO)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
                                  num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.unet.parameters(), lr=LEARNING_RATE)

    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    start_epoch = 0

    if RESUME_TRAINING and os.path.exists(RESUME_CHECKPOINT_PATH):
        accelerator.print(f"Resuming training from checkpoint: {RESUME_CHECKPOINT_PATH}")
        checkpoint = torch.load(RESUME_CHECKPOINT_PATH, map_location='cpu')
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  # 从上一次的下一个 epoch 开始
    
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )
    accelerator.init_trackers("flickr8k_IntegratedGenerator")

    global_step = 0
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_t2i_gen_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} Training", 
                                  disable=not accelerator.is_main_process, position=0, leave=True)
        for step, batch in enumerate(train_progress_bar):
            if batch is None: continue
            optimizer.zero_grad()
            pixel_values = batch['pixel_values']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']

            t2i_gen_loss = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                                mode='train_image_generation')
            accelerator.backward(t2i_gen_loss)
            optimizer.step()
            lr_scheduler.step()

            total_t2i_gen_loss += t2i_gen_loss.item()
            global_step += 1

            if accelerator.is_main_process and (step + 1) % 50 == 0:
                 avg_t2i_gen_loss = total_t2i_gen_loss / (step + 1)

                 accelerator.print(
                     f"Epoch {epoch+1}/{EPOCHS}, Step {step+1}/{len(train_dataloader)}, "
                     f"T2I Gen Loss: {t2i_gen_loss.item():.4f}, Avg T2I Gen Loss: {avg_t2i_gen_loss:.4f}, "
                     f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
                 )
                 accelerator.log({
                     "train_t2i_gen_loss": t2i_gen_loss.item(),
                     "learning_rate": lr_scheduler.get_last_lr()[0]
                 }, step=global_step)

        ####VAL#####
        model.eval()
        total_val_t2i_gen_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                 if batch is None: continue
                 pixel_values = batch['pixel_values']
                 input_ids = batch['input_ids']
                 attention_mask = batch['attention_mask']

                 val_t2i_gen_loss = model(
                     pixel_values=pixel_values,
                     input_ids=input_ids,
                     attention_mask=attention_mask,
                     mode='train_image_generation'
                 )
                 total_val_t2i_gen_loss += val_t2i_gen_loss.item()

                 if accelerator.is_main_process and epoch % 5==0 and random.random() < 0.3:
                     try:
                          unwrapped_model = accelerator.unwrap_model(model)
                          generated_images = unwrapped_model(input_ids=input_ids[:2], attention_mask=attention_mask[:2], mode='generate_image')

                          output_dir = 'val_gen_imgs'
                          os.makedirs(output_dir, exist_ok=True)

                          accelerator.print("\n--- Example Generation ---")
                          for i in range(generated_images.shape[0]):
                                img_tensor = generated_images[i].cpu()
                                img_tensor = img_tensor.permute(1, 2, 0) * 255
                                img_np = img_tensor.byte().numpy()
                                img_pil = Image.fromarray(img_np, 'RGB')
                                filename = os.path.join(output_dir, f"epoch{epoch+1}_step{step+1}_img{i+1}.png")
                                img_pil.save(filename)
                                accelerator.print(f"Saved generated image {i+1} to {filename}")
                          accelerator.print("------------------------\n")
                     except Exception as e:
                          accelerator.print(f"Generation failed: {e}")


        avg_val_t2i_gen_loss = total_val_t2i_gen_loss / len(val_dataloader)

        accelerator.print(f"Epoch {epoch+1} Validation T2I Gen Loss: {avg_val_t2i_gen_loss:.4f}")

        if accelerator.is_main_process:
             accelerator.log({
                 "val_t2i_gen_loss": avg_val_t2i_gen_loss,
             }, step=global_step)

             accelerator.wait_for_everyone()
             unwrapped_model = accelerator.unwrap_model(model)
             accelerator.save({
                    "epoch": epoch,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    "train_t2i_gen_loss": avg_t2i_gen_loss,
                    "val_t2i_gen_loss": avg_val_t2i_gen_loss,
                }, SAVE_CHECKPOINT_PATH)
             accelerator.print(f"Checkpoint saved to {SAVE_CHECKPOINT_PATH}")

    accelerator.end_training()
    print("Training finished.")


if __name__ == "__main__":
    main()