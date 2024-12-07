import csv
import pandas as pd
import os
import cv2
import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms as tr
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from Imgtxt_embeddings import *
from Image_with_CLIP import *
from basic_analysis import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)


def create_dataset(frames_folder, output_csv, labels=None):
    #df = pd.DataFrame(columns=["frame", "image_embedding", "brightness", "contrast", "scene_type", "emotion", "character", "description", "color"])
    movies = sorted([os.path.join(frames_folder, folder) for folder in os.listdir(frames_folder)])

    with open(output_csv, 'a', buffering=1) as csvfile:
        csvwriter = csv.writer(csvfile)

        for j in range(len(movies)):
            movie = movies[j]
            frame_files = [os.path.join(movie, f) for f in os.listdir(movie) if f.endswith("_img_1.jpg")]
            #randomly pick 256 frames from frame_files
            frame_files = np.random.choice(frame_files, size=256, replace=False)
            print(f"Processed movie {movie}, this is the {j+1} movie")
            for i, frame_file in enumerate(frame_files):
                #Extract image embeddings and other features
                image = cv2.imread(frame_file)
                image_embedding = imageEmbeddings(image, device, processor, model).get_embeddings()
                brightness, contrast = brightness_contrast(image)
                scene_type,emotion,characters,description = get_descriptions(frame_file)
                scene_type = textEmbeddings(scene_type, device, tokenizer, model).get_embeddings()
                emotion = textEmbeddings(emotion, device, tokenizer, model).get_embeddings()
                characters = textEmbeddings(characters, device, tokenizer, model).get_embeddings()
                description = textEmbeddings(description, device, tokenizer, model).get_embeddings()
                color,color_text = extract_colors(frame_file)
                csvwriter.writerow([os.path.join(os.path.basename(movie), os.path.basename(frame_file)), image_embedding, brightness, contrast, scene_type, emotion, characters, description, color])
                #df.loc[len(df)] = [image_embedding, brightness, contrast, scene_type, emotion, characters, description, color]
    return
    # Write to CSV
    # df.to_csv('output.csv', index=False)