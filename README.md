# Dynamic Images for Action Recognition

Python implementation of technology discussed in 'Dynamic Image Networks for Action Recognition' by Bilen et al.
Their paper and GitHub can be found here: https://github.com/hbilen/dynamic-image-nets

If you're planning on using this, please verify the correctness of the implementation for your own inputs and outputs!

# Installation

Clone the directory, and install the requirements specified in the "requirements.txt" file.
~~~~
pip install -r requirements.txt
~~~~

# Example Usage
~~~~
import glob
import cv2
import numpy as np
import dynamicimage


def main():
    frames = glob.glob('./example_frames/*.jpg')
    frames = [cv2.imread(f) for f in frames]

    dyn_image = dynamicimage.get_dynamic_image(frames, normalized=True)
    cv2.imshow('', dyn_image)
    cv2.waitKey()


if __name__ == '__main__':
    main()
~~~~

# Example Output
![Dynamic Image Example](dynamic_image_example.JPG)