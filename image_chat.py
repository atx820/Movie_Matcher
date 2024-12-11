from openai import OpenAI
import openai
from Image_with_CLIP import *
from basic_analysis import *
from PIL import Image
from keys import openai_api_key
import requests
import base64

api_key=openai_api_key
client = OpenAI(
    api_key=openai_api_key
)

def prepare_image(input_path, output_path):
    # Open the image
    image = Image.open(input_path)
    
    # Convert to RGBA format
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    # Save the converted image
    image.save(output_path, format="PNG")  # Save as PNG because DALL-E requires it

    return output_path

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
  
def modify_image(client, image_path, description, output_path):

    converted_path = prepare_image(image_path, "temp_img.png")
    response = client.images.edit(
        image = open(converted_path, "rb"),
        prompt= description,
        n=1,
        size="1024x1024",
        )
    
    # Extract the image URL from the response
    image_url =  response.data[0].url

    # Download and save the image
    img_data = requests.get(image_url).content
    with open(output_path, 'wb') as f:
        f.write(img_data)

    print(f"Image saved to {output_path}")

def generate_image(client, description, output_path):
    """
    Generates an image using the OpenAI client and saves it to a file.
    """
    # Generate the image using DALL-E

    response = client.images.generate(
        model="dall-e-3",
        prompt= description,
        n=1,
        size="1024x1024",
        style = "natural"
        )

    # Extract the image URL from the response
    image_url =  response.data[0].url

    # Download and save the image
    img_data = requests.get(image_url).content
    with open(output_path, 'wb') as f:
        f.write(img_data)

    print(f"Image saved to {output_path}")

def generate_chat(client,prompt):

    # Call the OpenAI API to generate the task
    response = client.chat.completions.create(
        model="gpt-4o",
        messages= [
            {"role" : "user", 
             "content" : prompt}
            ],
        max_tokens=350,
        temperature=0.5,
    )

    # Extract the generated task from the response
    response = response.choices[0].message.content

    return response

def generate_chat_img(client,image_path,prompt):

    # Getting the base64 string
    base64_image = encode_image(image_path)
    # Call the OpenAI API to generate the task
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                ],
            }
        ],
        max_tokens=200,
        temperature=0.3,
    )

    # Extract the generated task from the response
    response = response.choices[0].message.content

    return response

# Load the image
image_path = "Processed_frames/tt0049730/shot_0083_img_2.jpg"
image = cv2.imread(image_path)
file_name = os.path.basename(image_path)
name_without_ext, ext = os.path.splitext(file_name)

edit_image_path = "sample_images/test_edit.png"
edit_image = cv2.imread(image_path)

# Calling CLIP for textural info
type,emotion,character,description = get_descriptions(image_path)

# Analysing basic info such as colors in the image
color,color_text = extract_colors(image_path)

brightness,contrast = brightness_contrast(image)
bc_text = f"the brightness of this image is:{brightness}, the contrast of this image is:{contrast} "
print(bc_text)

test_prompt = f"""
Based on the following description below, can you guess which movie this is or suggest some movies that might match its aesthetic feelings?\n
{description}, {color_text}.
"""
response = generate_chat(client, test_prompt)
print(response)

test_img_prompt = "Can you descript what is inside this frame of a movie? Also please give some acurate introduction of it's dominant colors, brightness, saturation and contrast, and possible filter being used"
response_img = generate_chat_img(client, image_path, test_img_prompt)
print(response_img)
prompt_image = f"""{description}, {response_img}"""

#modify_image = modify_image(client, edit_image_path, response_img, "test_img_edit.png")


generated_image_name = f"processed_{name_without_ext}{ext}"  
generate_image(client, prompt_image, generated_image_name)

test_prompt = f"""
Based on the following description below, can you guess which movie this is or suggest some movies that might match its aesthetic feelings?\n
{description}, {color_text}, {response_img}.
"""
response = generate_chat(client, test_prompt)
print(response)