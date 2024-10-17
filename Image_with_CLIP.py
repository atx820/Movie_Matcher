import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def get_best_clip_match(image_path, descriptions):
    """
    Uses CLIP to find the best matching description for an image.
    
    Parameters:
        image_path (str): Path to the image file.
        descriptions (list): List of text descriptions to match against the image.
        
    Returns:
        str: The best matching description.
    """
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the image
    image = Image.open(image_path)

    # Pre-process the image and descriptions
    inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True)

    # Perform inference
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-text similarity
    probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    # Get the best-matching description
    best_match_idx = torch.argmax(probs).item()
    return descriptions[best_match_idx]

def get_descriptions(image_path):

    # Prepare text descriptions (you can modify these as needed)
    descriptions = [
        "scene of a sci-fi movie",
        "scene of a romantic movie",
        "scene  of a superhero movie",
        "scene of a art film",
        "scene  of a documentary",
        "scene of a historical movie",
        "scene of a horror movie",
        "scene of a action/adventure movie",
        "scene of a fantasy movie",
        "scene of a comedy movie",
    ]

    emotions = [
        "a peaceful scene",
        "a dramatic scene",
        "a melancholic mood",
        "an exciting atmosphere",
        "a mysterious setting", 
        "a epic/grand scene",
        "a romantic scene",
        "a tense/thrilling atmosphere",
        "a dark/blooding scene",
        "a chaotic/frantic setting"
    ]

    characters = [
        "solo character(single person scene)",
        "two characters(dual interaction)",
        "three characters(trangular dynamics)",
        "group scene(four or more characters)",
        "no characters(environment only)"
    ]

    best_match_type = get_best_clip_match(image_path, descriptions)
    best_match_emotion = get_best_clip_match(image_path, emotions)
    best_match_character = get_best_clip_match(image_path, characters)
    description = f"This is a {best_match_type}, There is/are {best_match_character}, The mood description of it is {best_match_emotion}"
    
    print (description)

    return (best_match_type, best_match_emotion, best_match_character, description)