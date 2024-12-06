import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as tr
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
#from torchvision.models import vit_h_14


class imageEmbeddings:
    """Class tasked with extracting image embeddings """
    def __init__(self, image, device, processor, model):
        self.image = image
        self.device = device
        self.processor = processor
        self.model = model


    # def process_test_image(self, image_path):
    #     """Processing images
    #     Parameters
    #     ----------
    #     image_path :str

    #     Returns
    #     -------
    #     Processed image : str
    #     """
    #     img = Image.open(image_path)
    #     transformations = tr.Compose([tr.ToTensor(),
    #                                     tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #                                     tr.Resize((518, 518))])
    #     img = transformations(img).float()
    #     img = img.unsqueeze_(0)

    #     img = img.to(self.device)

    #     return img

    def get_embeddings(self):
        #img = self.process_test_image(self.image_path_1)
        img = self.image
        processed_img = self.processor(text = None, images=img, return_tensors="pt")["pixel_values"].to(self.device)
        embedding = self.model.get_image_features(processed_img)
        embedding_as_np = embedding.cpu().detach().numpy()

        return embedding_as_np
    
class textEmbeddings:
    def __init__(self, text, device, tokenizer, model):
        self.text = text
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

    def get_embeddings(self):
        inputs = self.tokenizer(self.text, return_tensors = "pt").to(self.device)
        text_embeddings = self.model.get_text_features(**inputs)
        embedding_as_np = text_embeddings.cpu().detach().numpy()

        return embedding_as_np