import torch
import os
import sys
from PIL import Image
from transformers import BertTokenizerFast, ViTFeatureExtractor
try:
    from torchvision import transforms
    torchvision_available = True
except ImportError:
    torchvision_available = False

from model import MultimodalTransformer

VIT_MODEL = 'google/vit-base-patch16-224-in21k'
TEXT_MODEL = 'bert-base-uncased'
MAX_LENGTH = 64
SAVE_CHECKPOINT_PATH = "ckpt/UniGen.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_OUTPUT_DIR = "generated_output"

def load_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_image(image, feature_extractor):
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values'].to(DEVICE)

def preprocess_text(text, tokenizer, max_length):
    inputs = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    return inputs['input_ids'].to(DEVICE), inputs['attention_mask'].to(DEVICE)

def main():
    print(f"Using device: {DEVICE}")

    tokenizer = BertTokenizerFast.from_pretrained(TEXT_MODEL)
    if tokenizer.bos_token is None: tokenizer.add_special_tokens({'bos_token': '[SOS]'})
    if tokenizer.eos_token is None: tokenizer.add_special_tokens({'eos_token': '[EOS]'})

    feature_extractor = ViTFeatureExtractor.from_pretrained(VIT_MODEL)

    model = MultimodalTransformer(
        vit_model_name=VIT_MODEL,
        text_model_name=TEXT_MODEL,
        d_model=768,
    )
    model.text_decoder.resize_token_embeddings(len(tokenizer))
    model.text_encoder.resize_token_embeddings(len(tokenizer))

    try:
        checkpoint = torch.load(SAVE_CHECKPOINT_PATH, map_location=DEVICE)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded successfully from {SAVE_CHECKPOINT_PATH}")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {SAVE_CHECKPOINT_PATH}")
        return
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return

    model.to(DEVICE)
    model.eval()

    if not os.path.exists(DEFAULT_OUTPUT_DIR):
        os.makedirs(DEFAULT_OUTPUT_DIR)
    
    img_counter = 0

    print("\nInteractive Multimodal Generation")
    print("Enter text to generate an image, or an image path to generate a caption.")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        user_input = input("\nEnter text or image path: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            break

        is_image_path = False
        if os.path.exists(user_input) and user_input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
             is_image_path = True

        with torch.no_grad():
            if is_image_path:
                print(f"Input detected as image path: {user_input}")
                image = load_image(user_input)
                if image:
                    pixel_values = preprocess_image(image, feature_extractor)
                    try:
                        generated_caption = model.generate_caption(pixel_values, tokenizer, max_length=MAX_LENGTH)
                        print("\nGenerated Caption:")
                        for caption in generated_caption:
                            print(f"- {caption}")
                    except AttributeError:
                        print("Error: The loaded model does not support 'generate_caption'.")
                    except Exception as e:
                        print(f"Error during caption generation: {e}")

            elif user_input:
                print(f"Input detected as text: \"{user_input}\"")
                if not torchvision_available:
                     print("Error: torchvision is required for saving generated images, but it's not installed.")
                     continue

                input_ids, attention_mask = preprocess_text(user_input, tokenizer, MAX_LENGTH)
                try:
                    model.generate_image(img_counter,input_ids, attention_mask)

                except AttributeError:
                    print("Error: The loaded model does not support 'generate_image_from_text'.")
                except NotImplementedError:
                    print("Error: Text-to-image generation is not implemented in this model.")
                except Exception as e:
                    print(f"Error during image generation: {e}")
            
            else:
                 print("Input is empty. Please enter text or an image path.")


    print("Exiting program.")

if __name__ == "__main__":
    main()