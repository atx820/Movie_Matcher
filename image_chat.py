from openai import OpenAI
from Image_with_CLIP import *
from basic_analysis import *
from PIL import Image
from keys import openai_api_key


client = OpenAI(
    api_key=openai_api_key
)

def generate_chat(prompt):

    # Call the OpenAI API to generate the task
    response = client.chat.completions.create(
        model="gpt-4",
        messages= [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role" : "user", "content" : prompt}
            ],
        max_tokens=300,
        temperature=0.7,
    )

    # Extract the generated task from the response
    response = response.choices[0].message.content

    return response


# Load the image
image_path = "extracted_frames/tt0914798/shot_0081_img_1.jpg"

# Calling CLIP for textural info
type,emotion,character,description = get_descriptions(image_path)

# Analysing basic info such as colors in the image
color,color_text = extract_colors(image_path)


test_prompt = f"""
Based on the following description below, can you guess which movie this is or suggest some movies that might match its aesthetic feelings?\n
{description}, {color_text}.
"""

response = generate_chat(test_prompt)

print(response)