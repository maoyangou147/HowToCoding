import os
import glob
import cv2


def save_frame_every_n(video_file="F:/dwz/192.168.1.30/D9_S20241018092737_E20241018104117.mp4", n=50):
    # Open the video file.
    video = cv2.VideoCapture(video_file)

    # Check if video opened successfully
    if not video.isOpened(): 
        print("Error opening video file")
        return None

    frame_number = 0
    image_number = 1

    while True:
        ret, frame = video.read()

        if not ret:
            break

        if frame_number % n == 0:            
            cv2.imwrite(f'F:/dwz/test_image/{image_number}.png', frame)
            image_number += 1

        frame_number += 1

    video.release()
    print('Finished processing video')


def get_frame_rate(video_file):
    # Open the video file.
    video = cv2.VideoCapture(video_file)

    # Check if video opened successfully
    if not video.isOpened(): 
        print("Error opening video file")
        return None

    # Get the frames per second (fps) of the video
    fps = video.get(cv2.CAP_PROP_FPS)

    return fps

def get_folder_frame_rate(directory="F:/dwz/192.168.1.30/"):
    # 检查目录是否存在
    if not os.path.exists(directory):
        print("The directory does not exist")
        return

    # 切换到指定目录
    os.chdir(directory)
    i = 0

    # 列出所有 MP4 文件并打印帧率
    for file in glob.glob("*.mp4"):
        i = i + 1
        video = cv2.VideoCapture(file)
        fps = video.get(cv2.CAP_PROP_FPS)
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_seconds = frames / fps
        duration_hours = duration_seconds / 3600
        print(f"File: {file}, Duration: {duration_hours} hours")
    
    print(i)

# Replace 'your_video.mp4' with the path to the video file you want to process.
save_frame_every_n()
