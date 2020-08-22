import numpy as np
import os
import matplotlib.path as mplPath
from utils import *
import cv2
import json

PATH_VIDEO = "../../videos"
PATH_TRACKING = '../../baseline_results/info_tracking'

PATH_RESULTS = '../../baseline_results/info_counting_v1'
PATH_SVIDEO = '../../baseline_results/vis_counting_results_v1'
VISUALIZED = True

def check_bbox_inside_with_roi(bbox, mask):
    #check if four point of bbox all in roi area
    is_inside = True

    x_tl = bbox[0]
    y_tl = bbox[1]
    x_br = bbox[2]
    y_br = bbox[3]

    for x in [x_tl, x_br]:
        if x <= 0 or x >= mask.shape[1]:
            return False

    for y in [y_tl, y_br]:
        if y <= 0 or y >= mask.shape[0]:
            return False

    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:
        if mask[v[1], v[0]] == 0:
            is_inside = False
            return is_inside

    return is_inside

def check_bbox_outside_with_roi(bbox, mask):
    x_tl = bbox[0]
    y_tl = bbox[1]
    x_br = bbox[2]
    y_br = bbox[3]

    is_outside = True

    vertexs = [[x_tl, y_tl], [x_tl, y_br], [x_br, y_tl], [x_br, y_br]]
    for v in vertexs:

        if v[1] < 0 or v[1] >= mask.shape[0] or v[0] < 0 or v[0] >= mask.shape[1]:
            
            continue 
        if mask[v[1], v[0]] == 255: #at least one point lies inside roi => false
            is_outside = False

            return is_outside

    return is_outside
    
def check_box_intersect_roi(box, roi_mask):
    check_inside = check_bbox_inside_with_roi(box, roi_mask)
    check_outside = check_bbox_outside_with_roi(box, roi_mask)

    return not (check_inside  or check_outside)

def merge_2_boxes(box_1, box_2):
    min_x_tl = min(box_1[0], box_2[0])
    min_y_tl = min(box_1[1], box_2[1])
    max_x_br = max(box_1[2], box_2[2])
    max_y_br = max(box_1[3], box_2[3])

    merged_box = [min_x_tl, min_y_tl, max_x_br, max_y_br]
    return merged_box

def region_based_assignment(track_instance_pos, mois_region):

    for mv_id in mois_region:
        cur_moi_region = mois_region[mv_id]
        # use the first mv id if pos lies inside region
        if cur_moi_region[track_instance_pos[1]][track_instance_pos[0]] == 255:
            return mv_id
    return -1

def calc_angle(vec1, vec2):

    L1 = np.sqrt(vec1.dot(vec1))
    L2 = np.sqrt(vec2.dot(vec2))

    if L1 == 0 or L2 == 0:
        return 90
    cos = vec1.dot(vec2)/(L1*L2)
    if cos > 1:
        return 90
    angle = np.arccos(cos) * 360/(2*np.pi)
    return angle

def vector_based_assignment(track_instance_vec, mois_mov_vec):
    min_angle = 360
    min_mv_id = 0 
    
    # print(track_instance_vec)
    # print(mois_mov_vec)
    for mois_id in mois_mov_vec:
        points = mois_mov_vec[mois_id]
    
        vec_moi = np.array([points[1][0] - points[0][0], points[1][1]- points[0][1]])
        vec_track = np.array([track_instance_vec[1][0] - track_instance_vec[0][0], track_instance_vec[1][1]- track_instance_vec[0][1]])

        
        angle = calc_angle(vec_moi, vec_track)
        # print("cal", angle)
        if angle < min_angle: 
            min_angle = angle
            min_mv_id = mois_id

    # print(min_angle)

    return min_mv_id

def process_each_track(track_instance, mois_mov_vec, roi_mask, mois_region):
    '''
        Assign movement and end frameid for one track_instance
    '''

    if len(track_instance['bbox']) < 2:
        # use moi region to determine
        # can only determine when it intersect with roi
        cur_c_box = track_instance['bbox'][0][1:]
        cur_fr_id = track_instance['bbox'][0][0]

        if check_box_intersect_roi(cur_c_box, roi_mask):
            cur_position = track_instance['tracklet'][0]
            mv_id = region_based_assignment(cur_position, mois_region)
            
            if mv_id == -1: # not lies inside any moi => reject
                return track_instance

            track_instance["mov_id"] = mv_id
            track_instance["recorded_fr_id"] = cur_fr_id

        # if not, ignore it 
        return track_instance
            

    else:
        # use moi vector to determine
        for tb_id, tb_box in enumerate(reversed(track_instance['bbox'])):
            prev_c_box = track_instance['bbox'][tb_id-1][1:]
            cur_c_box = track_instance['bbox'][tb_id][1:]
            cur_fr_id = track_instance['bbox'][tb_id][0]

            merge_box = merge_2_boxes(prev_c_box, cur_c_box)
            if check_box_intersect_roi(merge_box, roi_mask):
                # one more condition to start counting
                if check_bbox_outside_with_roi(cur_c_box, roi_mask) and check_bbox_inside_with_roi(prev_c_box, roi_mask):
                    
                    # vector calculation and movement assignment
                    x_tail = track_instance['tracklet'][0][1]
                    y_tail = track_instance['tracklet'][0][0]
                    x_head = track_instance['tracklet'][-1][1]
                    y_head = track_instance['tracklet'][-1][0]

                    mov_vec = [(y_tail, x_tail), (y_head, x_head)]
                    mv_id = vector_based_assignment(mov_vec, mois_mov_vec)
                    
                    # start to record frame id and assign movement
                    track_instance["mov_id"] = mv_id
                    track_instance["recorded_fr_id"] = cur_fr_id
                    break # finish assign 
            if tb_id == 1:
                break # track_instance is destroyed before moving to roi edge=>ignore

    return track_instance

def counting(vid_name, roi_mask, mois_mov_vec, mois_region):
    cam_name = vid_name[:-4]
    # load info tracking from tracker

    track_info = np.load(PATH_TRACKING + '/info_' + cam_name + '.npy', allow_pickle = True)
    tracks = {}

    for each_info in track_info:
        frameid = int(each_info[1])
        trackid = int(each_info[3])
        x1 = int(float(each_info[4]))
        y1 = int(float(each_info[5]))
        x2 = int(float(each_info[6]))
        y2 = int(float(each_info[7]))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        label = each_info[0]
        if trackid in tracks:
            tracks[trackid]['endframe'] = frameid
            tracks[trackid]['bbox'].append([frameid, x1, y1, x2, y2])
            tracks[trackid]['tracklet'].append([cx, cy])
        else:
            tracks[trackid] = {'startframe' : frameid,
                                'endframe' : frameid,
                                'label': label,
                                'bbox' : [[frameid, x1, y1, x2, y2]],
                                'tracklet' : [[cx, cy]]}
    res_count = []

    for trackid, track_instance in tracks.items():
        # if trackid != 121:
        #     continue
        res_track = process_each_track(track_instance, mois_mov_vec, roi_mask, mois_region)
        if "mov_id" in res_track and "recorded_fr_id" in res_track:
            mov_id = res_track["mov_id"]
            # print(mov_id)
            recorded_fr_id = res_track["recorded_fr_id"]
            label = track_instance["label"]
            res_count.append([recorded_fr_id+1, mov_id, label])
    
    res_count.sort(key=lambda x: x[0])
    print(len(res_count))
    np.save(PATH_RESULTS + '/info_' + cam_name, res_count)

if __name__ == "__main__":
    if not os.path.isdir(PATH_RESULTS):
        os.mkdir(PATH_RESULTS)

    if not os.path.isdir(PATH_SVIDEO):
        os.mkdir(PATH_SVIDEO)

    mois_region = load_moi()
    roi_mask = load_roi_mask()
    mois_mov_vec = load_movement_vector()
    for video_name in os.listdir(PATH_VIDEO):
        cam_name = video_name[:-4]
        if video_name == "cam_01.mp4": #video_name .endswith(".mp4") or video_name == "cam_1.mp4":
            results = counting(video_name, roi_mask[cam_name], mois_mov_vec[cam_name], mois_region[cam_name])
