import numpy as np
import os
import matplotlib.path as mplPath
from utils import *
import cv2
import json

path_vis = '../../baseline_results/vis_counting_results_v1'
path_count_result = '../../baseline_results/info_counting_v1'
path_video = "../../videos"

def vis_video(video_name, roi_list, moi_list):
    input = cv2.VideoCapture(path_video + '/' + video_name)
    width = int(input.get(3)) # get width
    height = int(input.get(4)) #get height
    output = cv2.VideoWriter(path_vis + '/' + video_name, 
                            cv2.VideoWriter_fourcc('M','J','P','G'), 
                            5.0, 
                            (width, height))
    cur_fr_idx = 1
    results_counting = np.load(os.path.join(path_count_result, "info_"+video_name[:-4]+".npy"))
    run_idx_res = 0

    text_summary = build_text_name_dict_v1(moi_list) # reimplement
    while (input.isOpened()):
        ret, frame = input.read()
        if not ret:
            break
        annotate_fr = draw_roi(roi_list, frame)
        annotate_fr = draw_moi_v1(annotate_fr, video_name[:-4])
        
        fr_id_records = []
        # extract counting information of current frame
        while int(results_counting[run_idx_res][0]) == cur_fr_idx:
            fr_id_records.append(results_counting[run_idx_res][1:])
            run_idx_res += 1

        # if has record
        if len(fr_id_records) != 0:
            for each_record in fr_id_records:
                mov_id = each_record[0]
                label = each_record[1]
                # update tex_summary
                text_summary[str(mov_id)]["class_"+str(int(float(label)))] += 1
        
        # draw text_summary
        annotate_fr = draw_text_summarize_v1(annotate_fr, text_summary, width, height)
        output.write(annotate_fr)
        if cur_fr_idx == 900:
            break
        cur_fr_idx += 1
    
    print(text_summary)
    input.release()
    output.release()

if __name__ == "__main__":
    roi_list = load_roi()
    for file_name in os.listdir(path_video):
        if file_name.endswith(".mp4"):
            list_moi_name = get_moi_name(file_name[:-4])
            if file_name == "cam_01.mp4":
                vis_video(file_name, roi_list[file_name[:-4]], list_moi_name)