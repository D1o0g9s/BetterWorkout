import logging
import time
import edgeiq
import cv2
import math
import imutils
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
print(top_arm_image_width)
top_arm_image_height = top_arm_image.shape[0]
current_top_image = None

top_y_start = 0
top_y_end = 0

top_x_start = 0
top_x_end = 0

NUM_BOT_IMAGES = 79
current_bot_image_index = 0
bot_arm_image = cv2.imread('./images/arm move 2/arm move_00000_000'+"{:0>2d}".format(current_bot_image_index)+'.png')
bot_arm_image_width = bot_arm_image.shape[1]
bot_arm_image_height = bot_arm_image.shape[0]
current_bot_dimensions = (0,0)
current_bot_image = None

bot_y_start = 0
bot_y_end = 0

bot_x_start = 0
bot_x_end = 0

def euclideanDistance(x1, y1, x2, y2) :
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

def angle(top_x, top_y, bot_x, bot_y):
    # Calculates the angle to rotate Top by
    return math.degrees(math.tan(abs(top_x - bot_x) / abs(top_y - bot_y)))

def showTopArm(frame):
    # Displays the current top image at the current top calculated location
    frame[top_y_start:top_y_end, top_x_start:top_x_end] = current_top_image
    return frame


def updateTopArmImageAndLocation(shoulder_x, shoulder_y, elbow_x, elbow_y): 
    # Calculates the current top image
    new_height = int(euclideanDistance(shoulder_x, shoulder_y, elbow_x, elbow_y))
    scale_ratio = new_height / top_arm_image_height
    new_width = int(scale_ratio * top_arm_image_width)

    if new_width <= 0:
        new_width = 1
    if new_height <= 0:
        new_height = 1

    new_dimensions = (new_width, new_height)

    new_image = cv2.resize(top_arm_image, new_dimensions)

    top_angle = angle(shoulder_x, shoulder_y, elbow_x, elbow_y)
    new_image = imutils.rotate(new_image, top_angle)
    global current_top_image 
    current_top_image = new_image

    # Calculate the offsets we need to add to the image when we overlay the new image on
    height_to_inc = (0.5 * new_width) * math.sin(top_angle)
    width_to_dec = (0.5 * new_width) * math.sin(top_angle) * math.tan(top_angle)

    ## Assumes increasing y is down, increasing x is right

    original_x_start = shoulder_x - (0.5 * new_width) 
    original_y_start = shoulder_y

    original_x_end = shoulder_x + (0.5 * new_width) 
    original_y_end = shoulder_y + new_height

    new_x_start = int(original_x_start + width_to_dec) # Subtract the small bit that we decremented because of the rotation
    new_y_start = int(original_y_start - height_to_inc) # Add the small bit that we incremented because of the rotation

    # Calculate the height of the center bulk pieces
    mid_width = new_height * math.sin(top_angle)
    mid_height = new_height * math.cos(top_angle)
    new_x_end = int(shoulder_x + mid_width + (0.5*new_width * math.cos(top_angle)))
    new_y_end = int(shoulder_y + mid_height + height_to_inc)

    global top_y_start
    global top_y_end

    global top_x_start
    global top_x_end

    top_y_start = new_y_start
    top_y_end = new_y_end

    top_x_start = new_x_start
    top_x_end = new_x_end

def showCurBotImage(frame):
    # Display the bottom image
    frame[bot_y_start:bot_y_end, bot_x_start:bot_x_end] = current_bot_image
    return frame

def updateBotArm():
    # Update the bot image
    global current_bot_image_index
    current_bot_image_index = (current_bot_image_index + 1) % NUM_BOT_IMAGES
    global bot_arm_image
    bot_arm_image = cv2.imread('./images/arm move 2/arm move_00000_000'+"{:0>2d}".format(current_bot_image_index)+'.png')
    print(current_bot_dimensions)
    new_image = cv2.resize(bot_arm_image, current_bot_dimensions)
    global current_bot_image
    current_bot_image = new_image

def updateBotArmImageAndLocation(wrist_x, wrist_y, elbow_x, elbow_y): 
    # Update the bottom image and location (doesn't actually display it though)
    new_width = int(euclideanDistance(wrist_x, wrist_y, elbow_x, elbow_y))
    new_height = int((new_width / bot_arm_image_width) * bot_arm_image_height)
    if new_width <= 0:
        new_width = 1
    if new_height <= 0:
        new_height = 1
    print("new width:", new_width, "new height:", new_height)
    new_dimensions = (new_width, new_height) 
    global current_bot_dimensions
    current_bot_dimensions = new_dimensions
    new_image = cv2.resize(bot_arm_image, current_bot_dimensions)
    global current_bot_image
    current_bot_image = new_image

    new_y_start = int(elbow_y - new_height)
    new_x_start = int(elbow_x)
    new_y_end = int(elbow_y)
    new_x_end = int(elbow_x + new_width)

    global bot_y_start
    global bot_x_start
    global bot_y_end
    global bot_x_end

    bot_y_start = new_y_start
    bot_x_start = new_x_start
    bot_y_end = new_y_end
    bot_x_end = new_x_end



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
                cnt += 1
                frame = video_stream.read()

                prevMillis = millis
                millis = int(round(time.time() * 1000))
                time_elapsed += millis - prevMillis

                # Grab updated image 
                if time_elapsed >= 33.3333:
                    time_elapsed = 0 
                    updateBotArm()

                # If there is a current image, display it! 
                if not (current_bot_image is None):
                    frame = showCurBotImage(frame)
                if not (current_top_image is None):
                    frame = showTopArm(frame)
                
                # Only calculate / update pose every 10 cycles 
                if cnt == 10:
                    results = pose_estimator.estimate(frame)
                    # Generate text to display on streamer
                    text = ["Model: {}".format(pose_estimator.model_id)]
                    text.append(
                            "Inference time: {:1.3f} s".format(results.duration))
                    
                    for ind, pose in enumerate(results.poses):
                        # Only process pose of person 1
                        if (ind == 0) :

                            right_wrist_y = pose.key_points["Right Wrist"][1]
                            right_wrist_x = pose.key_points["Right Wrist"][0]
                            right_elbow_y = pose.key_points["Right Elbow"][1]
                            right_elbow_x = pose.key_points["Right Elbow"][0]
                            right_shoulder_y = pose.key_points["Right Shoulder"][1]
                            right_shoulder_x = pose.key_points["Right Shoulder"][0]

                            updateTopArmImageAndLocation(right_shoulder_x, right_shoulder_y, right_elbow_x, right_elbow_y)
                            updateBotArmImageAndLocation(right_wrist_x, right_wrist_y, right_elbow_x, right_elbow_y)

                            user1_right_dict = dict()
                            user1_right_dict["wrist_y"] = right_wrist_y
                            user1_right_dict["elbow_y"] = right_elbow_y
                            user1_right_dict["shoulder_y"] = right_shoulder_y

                            user1_right_dict["wrist_x"] = right_wrist_x
                            user1_right_dict["elbow_x"] = right_elbow_x
                            user1_right_dict["shoulder_x"] = right_shoulder_x

                            print("right arm data")
                            print(user1_right_dict)

                            # left_wrist_y = pose.key_points["Left Wrist"][1]
                            # left_wrist_x = pose.key_points["Left Wrist"][0]
                            # left_elbow_y = pose.key_points["Left Elbow"][1]
                            # left_elbow_x = pose.key_points["Left Elbow"][0]
                            # left_shoulder_y = pose.key_points["Left Shoulder"][1]
                            # left_shoulder_x = pose.key_points["Left Shoulder"][0]

                            # user1_right_dict["timestamp"] = [millis]
                                
                            # user1_left_dict["timestamp"] = [millis]
                            # user1_left_dict["wrist_y"] = [left_wrist_y]
                            # user1_left_dict["elbow_y"] = [left_elbow_y]
                            # user1_left_dict["shoulder_y"] = [left_shoulder_y]

                            # user1_left_dict["wrist_x"] = [left_wrist_x]
                            # user1_left_dict["elbow_x"] = [left_elbow_x]
                            # user1_left_dict["shoulder_x"] = [left_shoulder_x]

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
                    cnt = 0
                else :
                    if not current_top_image:
                        frame = showTopArm(frame)
                streamer.send_data(results.draw_poses(frame), text)

                fps.update()

                if streamer.check_exit():
                    break
    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
