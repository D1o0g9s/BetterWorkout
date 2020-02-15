import logging
import time
import edgeiq

import pandas as pd 
"""
Use pose estimation to determine human poses in realtime. Human Pose returns
a list of key points indicating joints that can be used for applications such
as activity recognition and augmented reality.

Pose estimation is only supported using the edgeIQ container with an NCS
accelerator.
"""
SEC_TO_KEEP = 5
MILLIS_TO_KEEP = SEC_TO_KEEP*1000  
def main():
    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")
    pose_estimator.load(
            engine=edgeiq.Engine.DNN_OPENVINO,
            accelerator=edgeiq.Accelerator.MYRIAD)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()

    # Create left and right dataframes 
    user1_right_dict = dict()
    # user1_right_dict["timestamp"] = [0] # Time in millis
    # user1_right_dict["wrist_y"] = [0]
    # user1_right_dict["elbow_y"] = [0]
    # user1_right_dict["shoulder_y"] = [0]

    # user1_right_dict["wrist_x"] = [0] 
    # user1_right_dict["elbow_x"] = [0] 
    # user1_right_dict["shoulder_x"] = [0] 

    user1_left_dict = dict()
    # user1_right_dict["timestamp"] = [0] # Time in millis
    # user1_left_dict["wrist_y"] = [0]
    # user1_left_dict["elbow_y"] = [0]
    # user1_left_dict["shoulder_y"] = [0]

    # user1_left_dict["wrist_x"] = [0]
    # user1_left_dict["elbow_x"] = [0]
    # user1_left_dict["shoulder_x"] = [0]

    # user1_right_df = pd.DataFrame.from_dict(user1_right_dict)
    # #user1_right_df.set_index('timestamp', inplace=True)
    # user1_left_df = pd.DataFrame.from_dict(user1_left_dict)
    # #user1_left_df.set_index('timestamp', inplace=True)

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            cnt = 9

            # For debugging purposes
            # prevMillis=0
            # millis = int(round(time.time() * 1000))
            # time_elapsed = 0 

            # loop detection
            while True:
                cnt += 1
                frame = video_stream.read()
                if cnt == 10:
                    results = pose_estimator.estimate(frame)
                    # Generate text to display on streamer
                    text = ["Model: {}".format(pose_estimator.model_id)]
                    text.append(
                            "Inference time: {:1.3f} s".format(results.duration))
                    
                    for ind, pose in enumerate(results.poses):
                        # Only process pose of person 1
                        if (ind == 0) :
                            # prevMillis = millis
                            # millis = int(round(time.time() * 1000))
                            # time_elapsed += millis - prevMillis


                            # if time_elapsed > (1000*1) : 
                            #     time_elapsed = 0

                                right_wrist_y = pose.key_points["Right Wrist"][1]
                                right_wrist_x = pose.key_points["Right Wrist"][0]
                                right_elbow_y = pose.key_points["Right Elbow"][1]
                                right_elbow_x = pose.key_points["Right Elbow"][0]
                                right_shoulder_y = pose.key_points["Right Shoulder"][1]
                                right_shoulder_x = pose.key_points["Right Shoulder"][0]

                                left_wrist_y = pose.key_points["Left Wrist"][1]
                                left_wrist_x = pose.key_points["Left Wrist"][0]
                                left_elbow_y = pose.key_points["Left Elbow"][1]
                                left_elbow_x = pose.key_points["Left Elbow"][0]
                                left_shoulder_y = pose.key_points["Left Shoulder"][1]
                                left_shoulder_x = pose.key_points["Left Shoulder"][0]

                                user1_right_dict["timestamp"] = [millis]
                                
                                user1_right_dict["wrist_y"] = [right_wrist_y]
                                user1_right_dict["elbow_y"] = [right_elbow_y]
                                user1_right_dict["shoulder_y"] = [right_shoulder_y]

                                user1_right_dict["wrist_x"] = [right_wrist_x]
                                user1_right_dict["elbow_x"] = [right_elbow_x]
                                user1_right_dict["shoulder_x"] = [right_shoulder_x]


                                user1_left_dict["timestamp"] = [millis]
                                user1_left_dict["wrist_y"] = [left_wrist_y]
                                user1_left_dict["elbow_y"] = [left_elbow_y]
                                user1_left_dict["shoulder_y"] = [left_shoulder_y]

                                user1_left_dict["wrist_x"] = [left_wrist_x]
                                user1_left_dict["elbow_x"] = [left_elbow_x]
                                user1_left_dict["shoulder_x"] = [left_shoulder_x]

                                # user1_right_df2 = pd.DataFrame.from_dict(user1_right_dict)
                                # #user1_right_df2.set_index('timestamp', inplace=True)
                                # user1_left_df2 = pd.DataFrame.from_dict(user1_left_dict)
                                # #user1_left_df2.set_index('timestamp', inplace=True)
                                # user1_right_df = pd.concat([user1_right_df, user1_right_df2])
                                # user1_left_df = pd.concat([user1_left_df, user1_left_df2])

                                print("right arm data")
                                print(user1_right_dict)

                            # text.append("Person {}".format(ind))
                            # text.append('-'*10)
                            # text.append("Key Points:")
                            # for key_point in pose.key_points:
                            #     text.append(str(key_point))
                    cnt = 0

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
