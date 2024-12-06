import tarfile
import os
import cv2
import ffmpeg

directory = '240P'
output_directory = 'extracted_frames'
for filename in sorted(os.listdir(directory)):
    f = os.path.join(directory, filename)
    modified_f = os.path.join(output_directory, filename.replace('.tar', ''))
    with tarfile.open(f, 'r') as tar:
        if (os.path.isdir(modified_f)==False):
            tar.extractall('extracted_frames/')
        elif (len(tar.getmembers()) != len(os.listdir(modified_f))-1):
          tar.extractall('extracted_frames/')   