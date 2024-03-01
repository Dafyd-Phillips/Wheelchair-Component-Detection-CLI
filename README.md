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
predict [image/video directory]: enter a directory to a .jpg or .png image to generate an annotated version of the image & a text file describing the components found. These are saved in their respective local directories. Enter a directory to a .mp4 file to generate an annotated version. A temporary directory is created to store the individual frames to generate the annotated video, after which the directory & its contents are deleted. Videos are saved in the respective local directory.

display --image/--description/--video [file_no]: display an annotated image, its description, or a video file. Files are saved & named in sequential order, starting from "1", with image and text files being generated as a pair.

eg display --description 4: display the text description for image file 4.

delete --image/video [file_no]: delete a single file. Both images and their respective text descriptions are deleted at once.

clear: deletes every annotated file and discription.

quit: closes the program

# Future Developments
"Display" command - request an annotated image/video or text description to be displayed.

"Delete" or "Clear" commands - delete files that are no longer needed from the CLI.

Filetype accommodation - allowing a wider range of image/video filetypes to be input.

Other features - damage assessment, detection of more varied components, etc - are limited by the wheelchair image data available for training. Please be patient!
