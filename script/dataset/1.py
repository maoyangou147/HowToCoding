import os
import glob
import cv2

def save_frame_every_n(video_file="/home/bob/dataset/dwz_raw/ent_video1/D17_S20241103103001_E20241103113057.mp4", n=50):
    # Open the video file.
    video = cv2.VideoCapture(video_file)

    # Check if video opened successfully
    if not video.isOpened(): 
        print("Error opening video file")
        return None

    frame_number = 0
    basename = os.path.basename(video_file)
    filename_without_ext = os.path.splitext(basename)[0]
    shour=[ 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    smin= [ 0, 1, 6, 7,12,23, 25,25,26,31,37,41,44,48,50,53,55]
    ssec= [26,42,24,19,51,17, 13,36,10,57,26,45, 6,14,29,32,48]
    ehour=[ 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    emin= [ 0, 1, 6, 7,13,23, 25,25,26,32,37,42,44,48,50,53,55]
    esec= [33,51,33,25, 0,33, 17,55,12, 0,28,34,50,44,32,40,57]  

    scope = []

    for i in range(len(shour)):
        start = (shour[i] * 3600 + smin[i] * 60 + ssec[i]) * 25
        end = (ehour[i] * 3600 + emin[i] * 60 + esec[i]) * 25
        scope.append((start, end))


    while True:
        ret, frame = video.read()

        if not ret:
            break

        frame_number += 1
        flag=False
        for (start, end) in scope:
            if frame_number >= start and frame_number <= end:
                flag=True
                break
        if not flag:
            continue

        if frame_number % n == 0:            
            cv2.imwrite(f'/home/bob/dataset/dwz_raw/tmp/{filename_without_ext}_{frame_number}.png', frame,[cv2.IMWRITE_PNG_COMPRESSION, 9])

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

def get_folder_frame_rate(directory="/home/bob/dataset/dwz_raw/video"):
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
        print(f"File: {file}, Duration: {fps} hours")
    
    print(i)

# Replace 'your_video.mp4' with the path to the video file you want to process.
save_frame_every_n()
# get_folder_frame_rate("/home/bob/dataset/dwz_raw/video")




'''

    shour=[ 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    smin= [ 3, 5,22,30, 37,29, 56,42,47,52, 8,10,20,20,30,35,39,42]
    ssec= [34,57,45, 0, 30,32,  0,20,12,15,51,53,21,33,46,54,55,24]
    ehour=[ 0, 0, 0, 0,  0, 0,  1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    emin= [ 4,20,25,35, 55,29, 22,42,47,52, 8,10,20,20,30,35,40,42]
    esec= [27, 0,45, 0, 25,40,  0,25,16,20,59,56,25,48,52,59, 2,30]

    D8
    shour=[ 0, 0, 0, 0]
    smin= [ 1,15,18,28]
    ssec= [ 1,25, 8,35]
    ehour=[ 0, 0, 0, 0]
    emin= [ 1,15,19,28]
    esec= [22,37,19,44]

    D17
    shour=[ 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    smin= [ 0, 1, 6, 7,12,23, 25,25,26,31,37,41,44,48,50,53,55]
    ssec= [26,42,24,19,51,17, 13,36,10,57,26,45, 6,14,29,32,48]
    ehour=[ 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    emin= [ 0, 1, 6, 7,13,23, 25,25,26,32,37,42,44,48,50,53,55]
    esec= [33,51,33,25, 0,33, 17,55,12, 0,28,34,50,44,32,40,57]    
'''