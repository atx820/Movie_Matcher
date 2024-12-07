import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from collections import Counter

def my_description_list():
    descriptions = [
        # Action and Adventure Scenes
        "scene of a superhero movie", "scene of an epic adventure", "scene of a spy thriller",
        "scene of a car chase", "scene of a heist movie", "scene of a martial arts movie",
        "scene of a survival adventure", "scene of a treasure hunt", "scene of a jungle expedition",
        "scene of a daring rescue", "scene of a post-apocalyptic world", "scene of a war movie",
        "scene of a naval battle", "scene of an air combat movie", "scene of a Western movie", "scene of a pirate movie", 
        "scene of a historical war drama", "scene of an epic journey", "scene of an explosive finale",
        "scene of sea voyage", "scene of a desert survival", "scene of a hero's sacrifice",

        # Sci-Fi and Futuristic Scenes
        "scene of a sci-fi movie", "scene of a space exploration movie", "scene of a dystopian world",
        "scene of a futuristic city", "scene of an alien invasion", "scene of a robot uprising",
        "scene of a cyberpunk world", "scene of a virtual reality movie", "scene of a galactic war", "scene of an advanced AI thriller",
        "scene of a futuristic utopia", "scene of a terraforming mission", "scene of a cryogenic awakening",
        "scene of a biotechnological experiment", "scene of a planetary colonization",
        "scene of a black hole discovery", "scene of an interstellar mission", "scene of a space battle",  "scene of a first alien contact",

        # Fantasy and Magical Scenes
        "scene of a fantasy movie", "scene of a magical adventure", "scene of a mythological story",
        "scene of a hero's quest", "scene of a magical kingdom", "scene of an enchanted forest", 
        "scene of a mystical journey", "scene of a fairy tale", "scene of a cursed land",

        # Romantic Scenes
        "scene of a romantic movie", "scene of a romantic comedy", "scene of a love story",
        "scene of a historical romance", "scene of a tragic romance", 
        "scene of a first love", "scene of a romantic getaway", "scene of a romantic tragedy",

        # Comedy Scenes
        "scene of a comedy movie", "scene of a slapstick comedy", "scene of a dark comedy",
        "scene of a parody movie", "scene of a family-friendly comedy", "scene of a buddy comedy",
        "scene of a workplace comedy", "scene of a road trip comedy", "scene of a hilarious heist", 

        # Horror and Thriller Scenes
        "scene of a horror movie", "scene of a supernatural horror", "scene of a haunted house",
        "scene of a psychological thriller", "scene of a zombie apocalypse", "scene of a murder mystery",
        "scene of a slasher movie", "scene of a monster movie", "scene of a cursed object",
        "scene of a ghost story", "scene of a paranormal investigation", "scene of a creepy forest",
        "scene of a creature stalking its prey", "scene of a foggy graveyard", 

        # Historical and Period Scenes
        "scene of a historical movie", "scene of a period drama", "scene of a royal court",
        "scene of a medieval battle", "scene of an empire's fall", "scene of a historical romance",
        "scene of a samurai duel", "scene of an ancient myth", "scene of a pharaoh's tomb",
        "scene of a great discovery", "scene of an artist's masterpiece", "scene of a Renaissance festival",
        "scene of a colonial settlement", "scene of a revolutionary uprising", "scene of a historical speech",
        "scene of a medieval feast", "scene of a great fire", "scene of a legendary king's court",

        # Crime and Mystery Scenes
        "scene of a crime movie", "scene of a detective story", "scene of a police procedural",
        "scene of a courtroom drama", "scene of a true crime story", "scene of a heist movie",
        "scene of a noir thriller", "scene of a murder investigation", "scene of a psychological crime drama",
        "scene of a political conspiracy", "scene of a prison escape", 

        # Sports Scenes
        "scene of a sports movie", "scene of a championship game", "scene of a team training montage",
        "scene of a dramatic game-winning moment", "scene of a team celebrating after a win", "scene of a bitter defeat", 

        # Documentary Scenes
        "scene of a wildlife documentary", "scene of a nature documentary", "scene of a historical documentary",
        "scene of a true crime documentary", "scene of a cultural documentary", "scene of a biographical documentary",
        "scene of a scientific exploration", "scene of an environmental crisis", "scene of a space exploration documentary",
        "scene of a documentary about art", "scene of a groundbreaking discovery",

        # Musical and Dance Scenes
        "scene of a musical film", "scene of a dance movie", "scene of a concert performance",
        "scene of a Broadway musical", "scene of a historical musical", "scene of a romantic musical",
        "scene of a choir performance", 

    ]

    emotions = [
        # Peaceful and Serene
        "a peaceful scene", "a serene atmosphere", "a tranquil setting", "a reflective mood",
        "a calming environment", "a harmonious scene", "a soothing atmosphere", "a meditative moment",
        "a restful setting", "a scene of quiet contentment", "a still and undisturbed moment",
        "a relaxing environment", "a blissful scene", "a sense of balance and order", "a scene of gentle beauty",

        # Excitement and Adventure
        "an exciting atmosphere", "a thrilling scene", "an adventurous setting", "a suspenseful moment",
        "a high-energy atmosphere", "a scene of daring and risk", "a heart-pounding moment",
        "an electrifying atmosphere", "a bold and audacious scene", "a sense of exploration",
        "a pulse-pounding sequence", "an exhilarating moment", "a fast-paced environment",
        "a scene filled with anticipation", "a dynamic and energetic moment",

        # Mystery and Suspense
        "a mysterious setting", "a tense atmosphere", "an ominous mood", "a foreboding scene",
        "a scene filled with intrigue", "an unsettling atmosphere", "a moment of lurking danger",
        "a shadowy and enigmatic setting", "a feeling of impending doom", "a puzzling and ambiguous mood",
        "a secretive and covert scene", "a mood of apprehension", "a scene shrouded in uncertainty",
        "a cryptic and haunting atmosphere", "a moment of dark anticipation",

        # Heroic and Grand
        "an epic/grand scene", "a majestic setting", "a heroic moment", "a powerful atmosphere",
        "a scene of great triumph", "a feeling of courage and bravery", "a scene filled with noble intentions",
        "a commanding and awe-inspiring moment", "a story of resilience and victory", "a setting of monumental scale",
        "a feeling of invincibility", "a moment of selfless sacrifice", "a triumphant and victorious scene",
        "a story of overcoming great odds", "a grand and sweeping view",

        # Sadness and Melancholy
        "a melancholic mood", "a sorrowful scene", "a somber atmosphere", "a heartbreaking moment",
        "a sense of profound loss", "a wistful and longing mood", "a tearful and emotional scene",
        "a mood of quiet despair", "a scene of mourning and grief", "a tragic and poignant moment",
        "a setting of desolation and loneliness", "a mood of bittersweet memories", "a quiet and introspective moment",
        "a feeling of heavy sadness", "a solemn and reflective atmosphere",

        # Romantic and Intimate
        "a romantic scene", "an intimate atmosphere", "a loving mood", "a passionate setting",
        "a moment of deep connection", "a tender and caring atmosphere", "a heartfelt and emotional moment",
        "a warm and affectionate scene", "a dreamy and enchanting atmosphere", "a moment of undeniable chemistry",
        "a story of unspoken feelings", "a setting filled with love and devotion", "a scene of playful flirting",
        "a mood of undying loyalty", "a magical and captivating moment",

        # Fear and Horror
        "a terrifying scene", "a fearful atmosphere", "a spine-chilling mood", "an anxiety-inducing setting",
        "a moment of abject terror", "a setting filled with dread", "a story of pure horror",
        "a sinister and malevolent atmosphere", "a suspenseful and nerve-wracking moment",
        "a nightmarish and haunting mood", "a setting of overwhelming fear", "a moment of sheer panic",
        "a story of confronting primal fears", "a menacing and predatory mood", "a shadowy and oppressive atmosphere",

        # Chaos and Disorder
        "a chaotic/frantic setting", "a disordered scene", "a hectic atmosphere", "an out-of-control moment",
        "a story filled with confusion and turmoil", "a scene of overwhelming chaos", "a moment of total anarchy",
        "a frenzied and wild setting", "a mood of reckless abandon", "a setting of complete disorder",
        "a frantic and unrelenting pace", "a mood of barely controlled panic", "a story of desperate survival",
        "a moment of spiraling chaos", "a scene of uncontrollable events",

        # Joy and Celebration
        "a joyous atmosphere", "a celebratory mood", "a cheerful scene", "a playful moment",
        "a story filled with laughter and happiness", "a scene of unbridled joy", "a setting of communal celebration",
        "a moment of lighthearted fun", "a mood of carefree bliss", "a festive and jubilant atmosphere",
        "a heartwarming and delightful story", "a setting of shared happiness", "a scene of childlike wonder",
        "a moment of pure elation", "a mood of contented laughter",

        # Hope and Inspiration
        "a hopeful scene", "an inspirational atmosphere", "a scene filled with optimism", "a moment of resilience",
        "a story of overcoming adversity", "a setting of new beginnings", "a mood of determined perseverance",
        "a scene of unshakable faith", "a story of redemption and hope", "a setting of limitless possibilities",
        "a moment of profound encouragement", "a story of boundless potential", "a mood of unyielding determination",
        "a setting filled with promise", "a triumphant and uplifting scene"
    ]

    characters = [
        # Solo Character Configurations (15)
        "solo character (single person scene)",
        "close-up of a solo character",
        "extreme close-up of a single character's face",
        "wide shot of a single character in a vast space",
        "silhouette of a lone figure against the light",
        "single character centered in the frame",
        "solo character off-center in the frame",
        "character seen from behind",
        "character in profile view",
        "over-the-shoulder shot of a character",
        "solo character from a high angle (bird's eye view)",
        "solo character from a low angle (worm's eye view)",
        "character framed by a doorway or window",
        "solo character partially obscured by shadows",
        "character reflected in a mirror or surface",

        # Two-Character Configurations (15)
        "two characters (dual interaction)",
        "two characters standing side by side",
        "two characters facing each other",
        "two characters in profile facing outward",
        "two characters walking together",
        "two characters sitting across from each other",
        "two characters in a symmetrical composition",
        "one character in the foreground, one in the background",
        "two characters framed within a doorway",
        "two characters leaning toward each other",
        "two characters in a back-to-back pose",

        # Three-Character Configurations (15)
        "three characters (triangular arrangement)",
        "three characters forming a line",
        "three characters forming a triangle",
        "three characters in a balanced composition",
        "one character flanked by two others",
        "three characters in a stacked composition (foreground, middle, background)",
        "three characters spaced evenly across the frame",
        "two characters interacting while one observes",
        "three characters aligned diagonally",
        "three characters partially overlapping each other",
        "one central character with two on opposite sides",
        "three characters forming a curve",
        "three characters facing outward from the center",
        "three characters placed asymmetrically",
        "three characters from a top-down view",

        # Group Scenes (20)
        "group scene (four or more characters)",
        "a group of characters forming a circle",
        "a group in a loose, scattered arrangement",
        "a group lined up in a row",
        "a group creating depth through staggered positions",
        "a tightly packed group",
        "a group with one character prominently in focus",
        "a group symmetrically arranged",
        "a group in a V-shaped composition",
        "a group seen from an elevated angle",
        "a group viewed from a low angle",
        "a group fanning out in a semi-circle",
        "a group forming a straight diagonal line",
        "a cluster of characters with overlapping figures",
        "a group forming concentric circles",
        "a group with clear leader-follower dynamics",
        "a group placed in a spiral arrangement",
        "a group with significant negative space around them",
        "a group divided into two distinct halves",
        "a group of characters blending into the background",

        # No Characters (10)
        "no characters (environment only)",
        "empty foreground with focus on the background",
        "scene composed entirely of negative space",
        "a symmetrical environment with no inhabitants",
        "an environment with leading lines to an empty center",
        "an overhead shot of a desolate area",
        "a wide shot emphasizing emptiness",
        "a close-up of an uninhabited detail",
        "an off-center composition of an empty space",
        "an extreme long shot of a barren landscape"
    ]
    
    return descriptions,emotions,characters

def get_best_clip_matches(image_path, descriptions, top_n):
    """
    Uses CLIP to find the top N matching descriptions for a single category.
    
    Parameters:
        image_path (str): Path to the image file.
        descriptions (list): List of text descriptions to match against the image.
        top_n (int): Number of top matches to return.
        
    Returns:
        list: A list of tuples containing the top N matching descriptions and their probabilities.
    """
    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Pre-process the image and descriptions
    inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True)

    # Perform inference
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # Image-text similarity
    probs = logits_per_image.softmax(dim=1)  # Convert to probabilities

    # Get the top N matches
    top_probs, top_indices = torch.topk(probs, k=top_n, dim=1)

    # Map indices to descriptions and probabilities
    top_matches = [
        (descriptions[idx], prob.item())
        for idx, prob in zip(top_indices[0], top_probs[0])
    ]

    return top_matches

def get_best_clip_match(image_path, descriptions,):
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
    descriptions,emotions,characters = my_description_list()

    best_match_type = get_best_clip_match(image_path, descriptions)
    best_match_emotion = get_best_clip_match(image_path, emotions)
    best_match_character = get_best_clip_match(image_path, characters)
    description = f"This is a {best_match_type}, There is/are {best_match_character}, The mood description of it is {best_match_emotion}"
    
    print (description)

    return (best_match_type, best_match_emotion, best_match_character, description)


def get_descriptions_batch(folder_path):

    descriptions,emotions,characters = my_description_list()

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
