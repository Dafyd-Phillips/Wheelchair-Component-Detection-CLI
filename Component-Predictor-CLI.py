import cv2
import os
import shutil

import ultralytics
from ultralytics import YOLO
import csv

# class to contain the component data derived from the csv file
# ROLE: stores and passes component data to handler.
class ComponentDataContainer():
    # containers for component information, ordered by their position in the csv file
    # eg the 2nd entry in names corresponds to the 2nd entry in ids, and so on for every list
    part_names = []
    part_ids = []
    part_descriptions = []
    related_parts = []

    # extract data from the csv and store it in each container
    def readCSV(self, csv_directory):
        component_listings = []
        # transpose data into a list
        with open(csv_directory, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')

            for row in csv_reader:
                component_listings.append(row)
            
        # for every entry, extract the relevant information and add them to the containers
        for entry in range(1, len(component_listings)):
            self.part_names.append(component_listings[entry][1])
            self.part_ids.append(component_listings[entry][2])
            self.part_descriptions.append(component_listings[entry][3])
            self.related_parts.append(component_listings[entry][4])
    
    def __init__(self, csv_directory):
        self.readCSV(csv_directory)

    # retrieve component data from a given position in the containers
    # ideally, the list should correspond to the component classes
    def get_component_details(self, part_no):
        part_name = self.part_names[part_no]
        part_id = self.part_ids[part_no]
        description = self.part_descriptions[part_no]
        related_parts = self.related_parts[part_no]

        component_data = [part_name, part_id, description, related_parts]

        return component_data
    
    def get_list_of_components(self):
        return self.part_names


#ROLE: take an image directory and make a prediction on it, passing the results to the handler.
class PredictionHandler():
    model = None

    # containers for the most recent predictions made
    class_predictons = None
    box_predictions = None
    confidence_scores = None

    # separated from init to allow other models to be loaded, if needed
    def load_model(self, model_directory):
        self.model = YOLO(model_directory)

    # ideally pre-trained to avoid having to train it before the program proper runs
    def __init__(self, model_directory):
        self.load_model(model_directory)

    # pass an image - directory or 
    def predict(self, image_source):
        print("image_source:", image_source)
        if image_source.all() == None:
            return False
        
        latest_prediction = self.model(image_source)

        # obtain class predictions, box coordinates and confidence scores
        class_predicts = []
        box_predicts = []
        confidences = []
        # get list of bounding boxes
        for predict in latest_prediction:
            bboxes = predict.boxes

            # get individual box predictions & extract their data
            for box in bboxes:
                class_predicts.append(int(box.cls))
                box_predicts.append(box.xywhn)

                # extract the confidence score, then convert it to an int percentage
                # confidence is stored as (float, ) in the bounding box data
                confidence_score = box.conf
                confidence_score = float(confidence_score[0])
                confidence_score = int(confidence_score * 100)
                confidences.append(confidence_score)

        # save bounding box information
        self.class_predictons = class_predicts
        self.box_predictions = box_predicts
        self.confidence_scores = confidences

        return latest_prediction

    def predict_video_frame(self, frame):
        return self.model.track(frame, persist=True)

    def get_class_predicts(self):
        if self.class_predictons != None:
            return self.class_predictons
        else:
            return False
    
    def get_box_predicts(self):
        if self.box_predictions != None:
            return self.box_predictions
        else:
            return False
        
    def get_confidence_scores(self):
        if self.confidence_scores != None:
            return self.confidence_scores
        else:
            return False


class InputOutputMediator():
    # directories for both the csv and YOLOv8 weights
    data_directory = "./Components-List.csv"
    model_directory = "./weights/best.pt"

    def __init__(self):
        # load the csv and weight files into their respective objects
        self.component_container = ComponentDataContainer(self.data_directory)
        self.prediction_model = PredictionHandler(self.model_directory)

        # containers for the most immediately relevant information
        self.relevant_data = []
    
    def make_prediction(self, image_source):
        # clear relevant data list for a new set of predictions
        self.relevant_data = []

        # make a prediction and get the classes
        full_prediction = self.prediction_model.predict(image_source)
        current_predictions = self.prediction_model.get_class_predicts()
        print(current_predictions)

        # fill container list with immediately relevant class data
        for entry in current_predictions:
            entry_data = self.component_container.get_component_details(entry)
            self.relevant_data.append(entry_data)

        return full_prediction, current_predictions

    def predict_video_frame(self, frame):
        return self.prediction_model.predict_video_frame(frame)
    
    #retrieve predicted class names
    def get_class_names(self):
        class_names = []
        for component in self.relevant_data:
            class_names.append(component[0])
        return class_names
    
    def get_class_predicts(self):
        return self.prediction_model.get_class_predicts()

    def get_box_coordinates(self):
        return self.prediction_model.get_box_predicts()
    
    def get_confidence_scores(self):
        return self.prediction_model.get_confidence_scores()

    # retrieve predicted class data for display in the UI
    def get_component_data(self, position):
        return self.relevant_data[position]
    
    def get_list_of_components(self):
        return self.component_container.get_list_of_components()

    
# User Interface class: passes image/video directory input to the mediator & utilies prediction results for file generation
class CommandLineIO():
    mediator = InputOutputMediator()
    image_file_index = 1
    video_file_index = 1
    text_file_index = 1

# initialising: get the number of files in each folder.
    def __init__(self):
        # check if directories exist, create if not
        if not os.path.exists("./Image-Predictions"):
            os.makedirs("./Image-Predictions")
        if not os.path.exists("./Video-Predictions"):
            os.makedirs("./Video-Predictions")
        if not os.path.exists("./Image-Prediction-Descriptions"):
            os.makedirs("./Image-Prediction-Descriptions")

        # check number of files 
        for root, dir, files in os.walk("./Image-Predictions"):
            if files:
                latest_file = files[-1]
                latest_file_split = latest_file.split(".")

                file_no = latest_file_split[0]
                self.image_file_index += int(file_no)

        for root, dir, files in os.walk("./Video-Predictions"):
            if files:
                latest_file = files[-1]
                latest_file_split = latest_file.split(".")

                file_no = latest_file_split[0]
                self.image_file_index += int(file_no)
        
        for root, dir, files in os.walk("./Image-Prediction-Descriptions"):
            if files:
                latest_file = files[-1]
                latest_file_split = latest_file.split(".")

                file_no = latest_file_split[0]
                self.text_file_index += int(file_no)

    def image_prediction(self, dir):
        try:
            img = cv2.imread(dir)
        except:
            print("ERROR: This directory does not exist")
            return False

        print("Generating image prediction.....")
        prediction, prediction_data = self.mediator.make_prediction(img)
        annotated_image = prediction[0].plot()
        print("PREDICTION DATA: ", prediction_data)

        print("Generating annotated image in Image-Predictions.....")
        imagename = "./Image-Predictions/" + str(self.image_file_index) + ".png"
        cv2.imwrite(imagename, annotated_image)
        self.image_file_index += 1

        print("Generating prediction details in Image-Prediction-Descriptions.....")
        predict_list = self.mediator.get_list_of_components()
        class_names = self.mediator.get_class_names()
        component_info = []
        
        print(predict_list, class_names)

        print(len(prediction))
        for predict in range(0, len(prediction_data)):
            print("predict: ", predict)
            component_info.append(self.mediator.get_component_data(predict))

        textname = "./Image-Prediction-Descriptions/" + str(self.text_file_index) + ".txt"
        with open(textname, "w") as file:

            for component in component_info:
                for line in component:
                    new_line = line + "\n"
                    file.writelines(new_line)
                file.writelines("\n")

            file.close()
        self.text_file_index += 1

        print("Image prediction successful!")
        
    def video_prediction(self, dir):
        try:
            capture = cv2.VideoCapture(dir)
        except:
            print("ERROR: This directory does not exist")
            return False
        frame_no = 0

        if not os.path.exists("./Temp-Video-Frames"):
            os.makedirs("./Temp-Video-Frames")

        print("Predicting & capturing annotated video frames.....")
        capture.open
        while capture.isOpened():
            success, frame = capture.read()

            if success:
                results = self.mediator.predict_video_frame(frame)

                annotated_frame = results[0].plot()
                frame_name = "./Temp-Video-Frames/" + str(frame_no) + '.png'
                cv2.imwrite(frame_name, annotated_frame)
                frame_no += 1
            else:
                break

        capture.release()
    
        frame_list = os.listdir("./Temp-Video-Frames")

        print("Generating annotated video file in Video-Predictions.....")
        if frame_list != []:
            frame_list = sorted(frame_list, key=lambda x: int(os.path.splitext(x)[0]))

            for i in range(0, len(frame_list)):
                item = "./Temp-Video-Frames/" + str(frame_list[i])
                image = cv2.imread(item)
                frame_list[i] = image
        else:
            print("Video frames failed to generate!")

        video_filename = "./Video-Predictions/" + str(self.video_file_index) + ".mp4"
        new_video=cv2.VideoWriter(video_filename, cv2.VideoWriter.fourcc('M', 'P', '4', 'V'), 25, (720, 576))

        for frames in range(0, len(frame_list)):
            new_video.write(frame_list[frames])
        
        self.video_file_index += 1
        shutil.rmtree("./Temp-Video-Frames")

        print("Video prediction successful!")
    
    def display_image_prediction(self, file_no):
        print("Displaying image.....")
        dir = "./Image-Predictions/" + str(file_no) + ".png"
        annotated_image = cv2.imread(dir)

        cv2.imshow(str(dir), annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def display_image_description(self, file_no):
        dir = "./Image-Prediction-Descriptions/" + str(file_no) + ".txt"
        description = open(dir)

        for line in description:
            print(line.strip())

        description.close()

    def display_video_prediction(self, file_no):
        dir = "./Video-Predictions/" + str(file_no) + ".mp4"
        capture = cv2.VideoCapture(dir)

        while capture.isOpened():
            success, frame = capture.read()

            if success:
                cv2.imshow(dir, frame)
            
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        capture.release()
        cv2.destroyAllWindows()
            
    def delete_image(self, file_no):
        confirm = input("You are about to delete this image & text description associated with it. Proceed? ")

        if confirm.lower() == "y":
            image = "./Image-Predictions/" + str(file_no) + ".png"
            text = "./Image-Prediction-Descriptions/" + str(file_no) + ".txt"
            os.remove(image)
            os.remove(text)
        else:
            return None

    def delete_video(self, file_no):
        confirm = input("You are about to delete this video. Proceed? ")

        if confirm.lower() == "y":
            video = "./Video_Predictions/" + str(file_no) + ".mp4"
            os.remove(video)
        else:
            return None

    def clear_files(self):
        confirm = input("YOU ARE ABOUT TO DELETE ALL PREDICTIONS. Proceed? ")

        if confirm.lower() == "y":
            shutil.rmtree("./Image-Predictions")
            shutil.rmtree("./Image-Prediction-Descriptions")
            shutil.rmtree("./Video_Predictions")
        else:
            return None

        os.makedirs("./Image-Predictions")
        os.makedirs("./Image-Prediction-Descriptions")
        os.makedirs("./Video_Predictions")

#    def help_command(self, command):
#        if command == "predict":
#            print("Input a filepath to an image or video file.\nSupports jpg, png, and mp4 formats.")
#        elif command == "display":
#            print("Request an annotated image or video file (starting from 0).\n--image for images & descriptions, --videos for annotated videos.")
#        elif command == "delete":
#            print("Delete an image & associated text description or an annotated video (starting from 0).")
#        elif command == "clear":
#            print("Deleted all annotated files & descriptions.")

    def main_loop(self):
        quit_program = False

        while quit_program == False:
            cl_input = input("Awaiting input: ")
            split_input = cl_input.split(" ")
            command = split_input[0]

            match command:
                case "predict":
                    directory = split_input[1]
                    split_dir = directory.split(".")
                    filetype = split_dir[-1]

                    if filetype.lower() == 'jpg' or filetype.lower() == 'png':
                        self.image_prediction(directory)
                    elif filetype.lower() == 'mp4':
                        self.video_prediction(directory)
                    else:
                        print("Invalid path: ", directory)
                
                case "display":
                    file_requested = split_input[1]
                    file_no = split_input[2]

                    if file_requested.lower() == "--image":
                        self.display_image_prediction(file_no)
                    elif file_requested.lower() == '--video':
                        self.display_video_prediction(file_no)
                    elif file_requested.lower() == '--description':
                        self.display_image_description(file_no)
                    else:
                        print("Please specify what file you want to display\n--image, --desciption, or --video")
                
                case "delete":
                    file_requested = split_input[1]
                    file_no = split_input[2]

                    if file_requested.lower() == "--image":
                        self.delete_image(file_no)
                    elif file_requested.lower() == "--video":
                        self.delete_video(file_no) 
                    else:
                        print("Please specify what file you want to delete\n--image or --video")

                case "clear":
                    self.clear_files()

                case "quit":
                    quit_program = True
                
                case _:
                    print("Please enter a valid command")


if __name__ == "__main__":
    cl = CommandLineIO()
    cl.main_loop()
