import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import transforms as tr
from torchvision.models import vit_h_14


class cosineSimilarity:
    """Class tasked with comparing similarity between two images """

    def __init__(self,image_path_1, image_path_2, device=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.image_path_1 = image_path_1
        self.image_path_2 = image_path_2

    def model(self):
        """Instantiates the feature extracting model

        Parameters
        ----------
        model
        Returns
        -------
        Vision Transformer model object
        """
        wt = torchvision.models.ViT_H_14_Weights.DEFAULT
        model = vit_h_14(weights=wt)
        model.heads = nn.Sequential(*list(model.heads.children())[:-1])
        model = model.to(self.device)

        return model

    def process_test_image(self, image_path):
        """Processing images
        Parameters
        ----------
        image_path :str

        Returns
        -------
        Processed image : str
        """
        img = Image.open(image_path)
        transformations = tr.Compose([tr.ToTensor(),
                                        tr.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        tr.Resize((518, 518))])
        img = transformations(img).float()
        img = img.unsqueeze_(0)

        img = img.to(self.device)

        return img

    def get_embeddings(self):
        """Computer embessings given images

        Parameters
        image_paths : str
        Returns
        -------
        embeddings: np.ndarray
        """
        img1 = self.process_test_image(self.image_path_1)
        img2 = self.process_test_image(self.image_path_2)
        model = self.model()

        emb_one = model(img1).detach().cpu()
        emb_two = model(img2).detach().cpu()

        return emb_one, emb_two

    def compute_scores(self):
        """Computes cosine similarity between two vectors."""
        emb_one, emb_two = self.get_embeddings()
        scores = torch.nn.functional.cosine_similarity(emb_one, emb_two)

        return scores.numpy().tolist()