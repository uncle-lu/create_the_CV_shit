# created by uncle-lu
# video to shit jpg
# 2021.06.01

import cv2, glob, os

def frames(path, save_path):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    train = open(os.path.join(save_path, "train.txt"), "w")

    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        if ret == False:
            print("ERROR")
            break
        this_frame_path = os.path.join(save_path, "{0:04d}.jpg".format(i))
        cv2.imwrite(this_frame_path,frame)
        train.write("{} {} {}\n".format(this_frame_path, this_frame_path, this_frame_path))
        print("save {}".format(this_frame_path))

    train.close()
    return 

if __name__ == '__main__':
    video_path = "./main.mp4"
    save_path = "./data/"
    frames(video_path, save_path)