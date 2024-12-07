import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import os

def extract_colors(image_path, n_colors=5):
    # Load image and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Use KMeans to cluster the colors
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixel_values)
    
    # Get the dominant colors
    centers = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_
    
    
    # Display the colors
    plt.figure()
    plt.imshow([centers])
    plt.axis("off")
    plt.show()

    color_text = f"Dominant colors are:\n {centers}"
    print(color_text)

    return centers,color_text

def is_dark_or_light(color, dark_threshold=30, light_threshold=225):
    """Check if the color is too dark or too light based on brightness thresholds."""
    mean_brightness = np.mean(color)
    return mean_brightness < dark_threshold or mean_brightness > light_threshold

def quantize_color(color, bin_size=10):
    """Quantize the color to reduce variations by rounding each RGB channel to the nearest bin."""
    return tuple((np.array(color) // bin_size * bin_size).astype(int))

def extract_colors_batch(folder_path, common_colors=5, image_limit=3000, bin_size=10, dark_threshold=30, light_threshold=225):
    """
    Extracts the dominant colors from a batch of images and outputs the 5 most common colors
    across all images by majority vote.
    
    Parameters:
        folder_path (str): Path to the folder containing images.
        n_colors (int): Number of colors to extract from each image.
        image_limit (int): Number of images to process (default is 0000).
    
    Returns:
        List of tuples: The 5 most common colors in RGB format.
    """
    color_counter = Counter()

    # Get list of images
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images = images[:image_limit]  # Limit to specified number of images

    for image_file in images:
        # Load the image
        image_path = os.path.join(folder_path, image_file)
        centers,color_text = extract_colors(image_path,n_colors=20)

        # Update the color counter
        for color in centers:
            if not is_dark_or_light(color, dark_threshold=dark_threshold, light_threshold=light_threshold):
                quantized_color = quantize_color(color, bin_size=bin_size)
                color_counter[quantized_color] += 1

    # Get the 5 most common colors by majority vote
    most_common_colors = [color for color, _ in color_counter.most_common(common_colors)]
    
    """
    # Display the colors
    plt.figure()
    plt.imshow([most_common_colors])
    plt.axis("off")
    plt.show()
    """
    color_text = f"Dominant colors are:\n {most_common_colors}"
    #print(color_text)

    return most_common_colors,color_text

def brightness_contrast(image):
    if image is None:
        print("Error: Unable to load image.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        brightness = np.mean(image_rgb)
        contrast = np.std(image_rgb)

        return brightness, contrast
