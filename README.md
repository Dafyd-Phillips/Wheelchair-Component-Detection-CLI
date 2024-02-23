# Wheelchair-Component-Detection-CLI
Repository for the ongoing development of a wheelchair component detection program.

(This program makes use of Ultralytics' YOLOv8 neural network for image/video predictions)
https://docs.ultralytics.com/

# Requirements
Python 3.7 or higher

Libraries:
OpenCV2 (cv2)
os
shutil
ultralytics
csv

# Use
Image prediction: enter a directory to a .jpg or .png image to generate an annotated version of the image & a text file describing the components found. These are saved in their respective local directories.

Video prediction: enter a directory to a .mp4 file to generate an annotated version. A temporary directory is created to store the individual frames to generate the annotated video, after which the directory & its contents are deleted. Videos are saved in the respective local directory.

Type 'q' to exit the program.

Files generated are numbered sequentially, with image & text files being generated as pairs.

# Future Developments
"Display" command - request an annotated image/video or text description to be displayed.
"Delete" or "Clear" commands - delete files that are no longer needed from the CLI.
Filetype accommodation - allowing a wider range of image/video filetypes to be input.

Other features - damage assessment, detection of more varied components, etc - are limited by the wheelchair image data available for training. Please be patient!
