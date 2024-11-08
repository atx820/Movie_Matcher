import cv2
import os
import ffmpeg

#modify path and loop to save all videos
image_folder = 'extracted_frames/tt0914798'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

(
    ffmpeg
    #modify path
    .input('/path/to/jpegs/*.jpg', pattern_type='glob', framerate=25)
    .output('movie.mp4')
    .run()
)