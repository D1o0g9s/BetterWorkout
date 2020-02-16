import logging
import time
import edgeiq
import cv2
import math
import imutils
import numpy as np
import threading
"""
Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

Pose estimation is only supported using the edgeIQ container with an NCS
accelerator.
"""

# Get images and the original width / height
top_arm_image = cv2.imread('./images/top.png')
top_arm_image_width = top_arm_image.shape[1]
top_arm_image_height = top_arm_image.shape[0]
top_ratio = 10
current_top_dimensions = (top_arm_image_width//top_ratio, top_arm_image_height//top_ratio)
current_top_image = cv2.resize(top_arm_image, current_top_dimensions)

top_y_start = 0
#top_y_end = 0

top_x_start = 0
#top_x_end = 0
bot_ratio = 5
NUM_BOT_IMAGES = 79
current_bot_image_index = 0
bot_arm_image = cv2.imread('./images/arm move 2/arm move_00000_000'+"{:0>2d}".format(current_bot_image_index)+'.png')
bot_arm_image_width = bot_arm_image.shape[1]
bot_arm_image_height = bot_arm_image.shape[0]
current_bot_dimensions = (bot_arm_image_width//bot_ratio, bot_arm_image_width//bot_ratio)
current_bot_image = cv2.resize(bot_arm_image, current_bot_dimensions)

bot_y_start = 0
#bot_y_end = 0

bot_x_start = 0
#bot_x_end = 0

existingThread = None

def euclideanDistance(x1, y1, x2, y2) :
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def angle(top_x, top_y, bot_x, bot_y):
    # Calculates the angle to rotate Top by
    return math.degrees(math.tan(abs(top_x - bot_x) / (abs(top_y - bot_y) if abs(top_y - bot_y) != 0 else 1 )))

def showTopArm(frame):
    # Displays the current top image at the current top calculated location
    width = current_top_dimensions[0] #current_top_image.shape[1]
    height = current_top_dimensions[1] #current_top_image.shape[0]
    
    y_start = (top_y_start if ((top_y_start > 0) and (top_y_start < frame.shape[0])) else 0)
    y_end = top_y_start + height 
    y_end = (y_end if ((y_end > 0) and (y_end < frame.shape[0])) else 0)


    x_start = (top_x_start if ((top_x_start > 0) and (top_x_start < frame.shape[1])) else 0)
    x_end = top_x_start + width
    x_end = (x_end if ((x_end > 0) and (x_end < frame.shape[1])) else 0)
    print("height", height, "width", width)
    toPutImage = current_top_image[0:(y_end-y_start),0:(x_end-x_start)]
    toPutFrame = frame[y_start:y_end, x_start:x_end]
    
    out = np.where(toPutImage == [0, 0, 0], toPutFrame, toPutImage)

    frame[y_start:y_end, x_start:x_end] = out #toPutImage
    return frame


def updateTopArmImageAndLocation(shoulder_x, shoulder_y, elbow_x, elbow_y): 
    # Calculates the current top image
    # new_height = int(euclideanDistance(shoulder_x, shoulder_y, elbow_x, elbow_y))
    # scale_ratio = new_height / top_arm_image_height
    # new_width = int(scale_ratio * top_arm_image_width)

    # if new_width <= 0:
    #     new_width = 1
    # if new_height <= 0:
    #     new_height = 1

    top_y = int(shoulder_y)
    top_x = int(shoulder_x - (0.5 * current_top_dimensions[0]) )
    if (elbow_y < shoulder_y) : 
        return
    if (elbow_x < shoulder_x) : 
        return 
    
    new_x_start = int(top_x) # Subtract the small bit that we decremented because of the rotation
    new_y_start = int(top_y) # Add the small bit that we incremented because of the rotation


    # new_dimensions = (new_width, new_height)

    #new_image = cv2.resize(top_arm_image, new_dimensions)
    #top_angle = -angle(shoulder_x, shoulder_y, elbow_x, elbow_y)
    # print("top angle", top_angle)
    #new_image = imutils.rotate_bound(new_image, top_angle)
    # global current_top_image 
    # current_top_image = new_image

    # Calculate the offsets we need to add to the image when we overlay the new image on
    #height_to_inc = abs((0.5 * new_width))# * math.sin(top_angle))
    #width_to_dec = abs((0.5 * new_width))# * math.sin(top_angle) * math.tan(top_angle))

    ## Assumes increasing y is down, increasing x is right

    # original_x_start = shoulder_x - (0.5 * new_width) 
    # original_y_start = shoulder_y

    # original_x_end = shoulder_x + (0.5 * new_width) 
    # original_y_end = shoulder_y + new_height

    
    # # Calculate the height of the center bulk pieces
    # mid_width = new_height * math.sin(top_angle)
    # mid_height = new_height * math.cos(top_angle)
    # new_x_end = int(shoulder_x + mid_width + (0.5*new_width * math.cos(top_angle)))
    # new_y_end = int(shoulder_y + mid_height + height_to_inc)

    global top_y_start
    #global top_y_end

    global top_x_start
    #global top_x_end

    top_y_start = new_y_start
    #top_y_end = new_y_end

    top_x_start = new_x_start
    #top_x_end = new_x_end

    print("top_y_start", top_y_start, "top_x_start", top_x_start) #"top_y_end", top_y_end, "top_x_end", top_x_end)

def showCurBotImage(frame):
    # Display the bottom image
    width = current_bot_image.shape[1]
    height = current_bot_image.shape[0]

    y_start = (bot_y_start if ((bot_y_start > 0) and (bot_y_start < frame.shape[0])) else 0)
    y_end = bot_y_start + height 
    y_end = (y_end if ((y_end > 0) and (y_end < frame.shape[0])) else 0)


    x_start = (bot_x_start if ((bot_x_start > 0) and (bot_x_start < frame.shape[1])) else 0)
    x_end = bot_x_start + width
    x_end = (x_end if ((x_end > 0) and (x_end < frame.shape[1])) else 0)

    #print("bot y_start:", y_start, "x_start", x_start, "y_end", y_end,  "x_end", x_end)
    toPutImage = current_bot_image[0:(y_end-y_start),0:(x_end-x_start)]
    toPutFrame = frame[y_start:y_end, x_start:x_end]

    out = np.where(toPutImage == [0, 0, 0], toPutFrame, toPutImage)

    frame[y_start:y_end, x_start:x_end] = out
    return frame

def updateBotArm():
    # Update the bot image
    global current_bot_image_index
    current_bot_image_index = (current_bot_image_index + 1) % NUM_BOT_IMAGES
    global bot_arm_image
    bot_arm_image = cv2.imread('./images/arm move 2/arm move_00000_000'+"{:0>2d}".format(current_bot_image_index)+'.png')
    new_image = cv2.resize(bot_arm_image, current_bot_dimensions)
    global current_bot_image
    if not(current_bot_image is None) :
        current_bot_image = new_image

def updateBotArmImageAndLocation(wrist_x, wrist_y, elbow_x, elbow_y): 
    # Update the bottom image and location (doesn't actually display it though)


    # new_width = int(euclideanDistance(wrist_x, wrist_y, elbow_x, elbow_y))
    # new_height = int((new_width / bot_arm_image_width) * bot_arm_image_height)
    # if new_width <= 0:
    #     new_width = 1
    # if new_height <= 0:
    #     new_height = 1

    bottom_y = int(top_y_start + current_top_image.shape[0] - current_bot_dimensions[1])
    bottom_x = int(top_x_start + current_top_image.shape[1])
    if (wrist_y > elbow_y) : 
        return
    if (wrist_x < elbow_x) : 
        return 
    
    # new_dimensions = (new_width, new_height) 
    # global current_bot_dimensions
    # current_bot_dimensions = new_dimensions
    
    # new_y_start = int(elbow_y - new_height)
    # new_x_start = int(elbow_x)
    #new_y_end = int(elbow_y)
    #new_x_end = int(elbow_x + new_width)

    global bot_y_start
    global bot_x_start
    #global bot_y_end
    #global bot_x_end

    bot_y_start = bottom_y
    bot_x_start = bottom_x
    #bot_y_end = new_y_end
    #bot_x_end = new_x_end

    print("bot_y_start", bot_y_start, "bot_x_start", bot_x_start) 

def estimatePoses(streamer, pose_estimator, frame):

    results = pose_estimator.estimate(frame)
    # Generate text to display on streamer
    text = ["Model: {}".format(pose_estimator.model_id)]
    text.append(
            "Inference time: {:1.3f} s".format(results.duration))
    
    for ind, pose in enumerate(results.poses):
        # Only process pose of person 1
        if (ind == 0) :

            # right_wrist_y = pose.key_points["Right Wrist"][1]
            # right_wrist_x = pose.key_points["Right Wrist"][0]
            # right_elbow_y = pose.key_points["Right Elbow"][1]
            # right_elbow_x = pose.key_points["Right Elbow"][0]
            # right_shoulder_y = pose.key_points["Right Shoulder"][1]
            # right_shoulder_x = pose.key_points["Right Shoulder"][0]

            # user1_right_dict = dict()
            # user1_right_dict["wrist_y"] = right_wrist_y
            # user1_right_dict["elbow_y"] = right_elbow_y
            # user1_right_dict["shoulder_y"] = right_shoulder_y

            # user1_right_dict["wrist_x"] = right_wrist_x
            # user1_right_dict["elbow_x"] = right_elbow_x
            # user1_right_dict["shoulder_x"] = right_shoulder_x

            left_wrist_y = pose.key_points["Left Wrist"][1]
            left_wrist_x = pose.key_points["Left Wrist"][0]
            left_elbow_y = pose.key_points["Left Elbow"][1]
            left_elbow_x = pose.key_points["Left Elbow"][0]
            left_shoulder_y = pose.key_points["Left Shoulder"][1]
            left_shoulder_x = pose.key_points["Left Shoulder"][0]

            user1_dict = dict()
            user1_dict["wrist_y"] = left_wrist_y
            user1_dict["elbow_y"] = left_elbow_y
            user1_dict["shoulder_y"] = left_shoulder_y

            user1_dict["wrist_x"] = left_wrist_x
            user1_dict["elbow_x"] = left_elbow_x
            user1_dict["shoulder_x"] = left_shoulder_x

            skip = False
            for key in user1_dict.keys():
                if user1_dict[key] < 0 : 
                    skip = True
            if not skip: 
                updateTopArmImageAndLocation(user1_dict["shoulder_x"], user1_dict["shoulder_y"], user1_dict["elbow_x"], user1_dict["elbow_y"])
                updateBotArmImageAndLocation(user1_dict["wrist_x"], user1_dict["wrist_y"], user1_dict["elbow_x"], user1_dict["elbow_y"])

            print("one arm data")
            print(user1_dict)

            # user1_right_df2 = pd.DataFrame.from_dict(user1_right_dict)
            # #user1_right_df2.set_index('timestamp', inplace=True)
            # user1_left_df2 = pd.DataFrame.from_dict(user1_left_dict)
            # #user1_left_df2.set_index('timestamp', inplace=True)
            # user1_right_df = pd.concat([user1_right_df, user1_right_df2])
            # user1_left_df = pd.concat([user1_left_df, user1_left_df2])

            

            # text.append("Person {}".format(ind))
            # text.append('-'*10)
            # text.append("Key Points:")
            # for key_point in pose.key_points:
            #     text.append(str(key_point))
    


def main():
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(
            engine=edgeiq.Engine.DNN_OPENVINO,
            accelerator=edgeiq.Accelerator.MYRIAD)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            cnt = 9

            # For debugging purposes
            prevMillis= 0
            millis = int(round(time.time() * 1000))
            time_elapsed = 0 

            # loop detection
            while True:
                time.sleep(0.05)
                cnt += 1
                frame = video_stream.read()

                prevMillis = millis
                millis = int(round(time.time() * 1000))
                time_elapsed += millis - prevMillis

                # Grab updated image 
                # if time_elapsed >= 33.3333:
                #     time_elapsed = 0 
                #     updateBotArm()

                # If there is a current image, display it! 
                if not (current_bot_image is None):
                    frame = showCurBotImage(frame)
                if not (current_top_image is None):
                    frame = showTopArm(frame)
                
                # Only calculate / update pose every 10 cycles 
                if cnt == 10:
                    cnt = 0
                    global existingThread
                    if not (existingThread is None):
                        existingThread.join()
                    existingThread = threading.Thread(target=estimatePoses, args=(streamer, pose_estimator,frame))
                    existingThread.start()
                
                streamer.send_data(frame, "")

                fps.update()

                if streamer.check_exit():
                    
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")
        
        # if not (existingThread is None):
        #     global existingThread
        #     existingThread.join()


if __name__ == "__main__":
    main()
