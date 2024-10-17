import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

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
    
    """
    # Display the colors
    plt.figure()
    plt.imshow([centers])
    plt.axis("off")
    plt.show()
    """
    color_text = f"Dominant colors are:\n {centers}"
    print(color_text)
    
    return centers,color_text

