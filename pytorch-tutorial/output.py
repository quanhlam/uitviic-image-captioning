import json
import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

from config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    try:
        image = Image.open(image_path)
        image = image.resize([224, 224], Image.LANCZOS)
        
        if transform is not None:
            image = transform(image).unsqueeze(0)
        
        return image
    except Exception as e:
         print(f"Error at image {image_path}; {e}")
         return image

def get_val_images(val_path):
    f = open(val_path, 'r+', encoding='utf8')
    data = json.load(f)
    return data['images']

def main(args):
    print("Initializing...")
    config = Config()
     # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])

    with open(config.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print("Vocabulary loaded")
    #Build models
    encoder = EncoderCNN(config.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(config.embed_size, config.hidden_size, len(vocab), config.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    print("Model built")
    encoder_path = args.encoder_path if args.encoder_path else config.encoder_path
    decoder_path = args.decoder_path if args.decoder_path else config.decoder_path
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    print("Model loaded")
    images = get_val_images(config.val_path)

    print("Valdation file loaded")
    results_data = []
    curr_id = 0
    for index, image_data in enumerate(images):
        try:
            image_path = config.img_path + "/" + image_data['file_name']
            image = load_image(image_path, transform)
            image_tensor = image.to(device)

            # Generate an caption from the image
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)

            sampled_caption = []

            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                if not word in ['<start>', '<end>']:
                    sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            sentence = sentence.replace('<start> ','')
            sentence = sentence.replace(' <end>', '')
            record = {
                'image_id': int(image_data['id']),
                'caption': sentence,
                'id': curr_id
            }
            curr_id+=1
            results_data.append(record)
            if index%10 == 0:
                print(f"Done image {index}/{len(images)}")
        except Exception as e:
            print(e)
            pass
    with open(config.machine_output_path, 'w+') as f_results:
        f_results.write(json.dumps(results_data, ensure_ascii=False))
    print("Finished")
    # Print out the image and the generated caption
    # print (sentence)
    # image = Image.open(path)
    # plt.imshow(np.asarray(image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=False, help='input image for generating caption', default=None)
    parser.add_argument('--encoder_path', type=str, required=False, help='encoder path', default=None)
    parser.add_argument('--decoder_path', type=str, required=False, help='decoder path', default=None)
    args = parser.parse_args()
    main(args)
