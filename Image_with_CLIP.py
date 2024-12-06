import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from collections import Counter

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
        "scene of a sci-fi movie", "scene of a futuristic sci-fi setting", "scene of a space exploration movie",
        "scene of a dystopian sci-fi world", "scene of a superhero movie", "scene of a fantasy movie",
        "scene of a magical adventure", "scene of a mythological story", "scene of an alternate reality",
        "scene of a romantic movie", "scene of a romantic comedy", "scene of a love story",
        "scene of a historical romance", "scene of a tragic romance", "scene of a coming-of-age romance",
        "scene of an action/adventure movie", "scene of a superhero action sequence", "scene of a spy thriller",
        "scene of a heist movie", "scene of a martial arts film", "scene of a car chase",
        "scene of an epic adventure journey", "scene of a survival movie",
        "scene of a dramatic movie", "scene of a family drama", "scene of a courtroom drama",
        "scene of a crime drama", "scene of a political drama", "scene of a social issues drama",
        "scene of a psychological thriller", "scene of a medical drama",
        "scene of a horror movie", "scene of a supernatural horror", "scene of a psychological thriller",
        "scene of a murder mystery", "scene of a haunted house", "scene of a monster movie",
        "scene of a slasher film", "scene of a zombie apocalypse", "scene of a crime thriller",
        "scene of a comedy movie", "scene of a romantic comedy", "scene of a dark comedy",
        "scene of a slapstick comedy", "scene of a parody film", "scene of a family-friendly comedy",
        "scene of a satire", "scene of a buddy comedy",
        "scene of a historical movie", "scene of a period drama", "scene of a medieval battle",
        "scene of a war movie", "scene of an ancient civilization", "scene of a biography film",
        "scene of a historical romance", "scene of a political war drama", "scene of a revolution story",
        "scene of a documentary", "scene of a wildlife documentary", "scene of a nature documentary",
        "scene of a biographical film", "scene of a true crime documentary", "scene of a historical documentary",
        "scene of a sports documentary", "scene of a cultural documentary",
        "scene of an animated movie", "scene of a family-friendly animation", "scene of a children's adventure",
        "scene of an animated fantasy", "scene of a musical animation", "scene of a superhero animation",
        "scene of a whimsical animated world", "scene of a family-oriented story",
        "scene of a musical film", "scene of a dance movie", "scene of a concert performance",
        "scene of a historical musical", "scene of a jukebox musical", "scene of a romantic musical",
        "scene of a classic musical"
    ]

    emotions = [
        "a peaceful scene", "a serene atmosphere", "a tranquil setting", "a reflective mood", 
        "an exciting atmosphere", "a thrilling scene", "an adventurous setting", "a suspenseful moment", 
        "a mysterious setting", "a tense atmosphere", "an ominous mood", "a foreboding scene", 
        "an epic/grand scene", "a majestic setting", "a heroic moment", "a powerful atmosphere", 
        "a melancholic mood", "a sorrowful scene", "a somber atmosphere", "a heartbreaking moment", 
        "a romantic scene", "an intimate atmosphere", "a loving mood", "a passionate setting", 
        "a dark/brooding scene", "a sinister mood", "a gloomy atmosphere", "a haunting setting", 
        "a chaotic/frantic setting", "a disordered scene", "a hectic atmosphere", "an out-of-control moment", 
        "a dramatic scene", "an intense confrontation", "a climactic moment", "a powerful emotional exchange", 
        "a joyous atmosphere", "a celebratory mood", "a cheerful scene", "a playful moment", 
        "a terrifying scene", "a fearful atmosphere", "a spine-chilling mood", "an anxiety-inducing setting", 
        "a hopeful scene", "an inspirational atmosphere", "a scene filled with optimism", "a moment of resilience", 
        "a perplexing scene", "an ambiguous mood", "a surreal atmosphere", "an unclear setting", 
        "a lonely scene", "a feeling of isolation", "an abandoned atmosphere", "a desolate mood"
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
    
    #print (description)

    return (best_match_type, best_match_emotion, best_match_character, description)


def get_descriptions_batch(folder_path):

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

    type_counter = Counter()
    emotion_counter = Counter()
    character_counter = Counter()

    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_path in image_paths:
        # Get the best match for each category in the current frame
        best_match_type = get_best_clip_match(image_path, descriptions)
        best_match_emotion = get_best_clip_match(image_path, emotions)
        best_match_character = get_best_clip_match(image_path, characters)

        # Update the counters
        type_counter[best_match_type] += 1
        emotion_counter[best_match_emotion] += 1
        character_counter[best_match_character] += 1

    # Get the most common result for each category by majority vote
    most_common_type = type_counter.most_common(1)[0][0]
    most_common_emotion = emotion_counter.most_common(1)[0][0]
    most_common_character = character_counter.most_common(1)[0][0]

    description = (
        f"This is a {most_common_type}. "
        f"There is/are {most_common_character}. "
        f"The mood description of it is {most_common_emotion}."
    )

    #print(description)

    return most_common_type, most_common_emotion, most_common_character, description
