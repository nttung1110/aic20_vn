import numpy as np
import pickle
import os
import json
import time
from iou_tracker import *
# path to video data
path_video = "../../videos"
# path to different detected bbox results
path_bbox = "../../bbox"

def format_bbox(video_name, file_name):
    ''' prepare formatted bbox for tracking'''
    with open(os.path.join(path_bbox, file_name), 'rb') as f_p:
        content = pickle.load(f_p)
        print("Processing:",video_name)
        data = []

        for fr_id, fr_content in enumerate(content):
            dets = []

            all_type_bboxes = [(fr_content[0], 1), (fr_content[1], 2), (fr_content[2], 3), (fr_content[3], 4)]
            for each_type_bboxes in all_type_bboxes:
                list_bboxes = each_type_bboxes[0]
                class_id = each_type_bboxes[1]
                for bb in list_bboxes:
                    dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': bb[4], 'class': class_id})
            data.append(dets)

    return data

if __name__ == "__main__":
    duration = time.time()
    print("Running tracking on ", path_bbox)
    for file_name in os.listdir(path_bbox):
        vid_name = file_name[:-4]
        data = format_bbox(vid_name, file_name)
        content_video_path = os.path.join(path_video, vid_name+".mp4")
        results = track_iou_edited(vid_name, data, 0.3, 0.7, 0.15, 20, path_video)
    duration = time.time() - duration 
    print("Total tracking time:", duration)
