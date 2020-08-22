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



def out_of_roi(center, poly, height):
    path_array = []
    for poly_point in poly:
        path_array.append([poly_point[1], height-poly_point[0]])
    path_array = np.asarray(path_array)
    polyPath = mplPath.Path(path_array)

    transform_center = (center[1], height-center[0])
    return polyPath.contains_point(transform_center, radius = 0.5)

def validate_center(center, use_off_set, roi_list, height):
    const = 5
    off_set = [(const, const), (const, -const), (-const, const), (-const, -const)]

    if not use_off_set:
        return  out_of_roi(center, roi_list, height)

    for each_off_set in off_set:
        center_change = (center[0] + each_off_set[0], center[1] + each_off_set[1])
        if out_of_roi(center_change, roi_list, height):
            return False
    return True

def out_of_range_bbox(tracking_info, width, height, off_set):
    x_min = int(tracking_info[4])
    y_min = int(tracking_info[5])
    x_max = int(tracking_info[6])
    y_max = int(tracking_info[7])
    return ((x_min-off_set <=0) or (y_min-off_set<=0) or (x_max+off_set>=width) or (y_max+off_set>=height))


def find_latest_object_and_vote_direction(frame_id_list, cur_fr_id, tracking_info, delta_fix, target_obj_id, roi_list, width, height):
    exist_latest_obj = False
    count_out = 0
    count_in = 0
    offset = 10
    for delta in range(1, delta_fix):
        pre_index = np.where(frame_id_list == (cur_fr_id - delta))[0]
        for each_pre_index in pre_index:
            if tracking_info[each_pre_index][3]==target_obj_id:
                exist_latest_obj = True
                pre_obj_center = center_box(tracking_info[each_pre_index][4:])
                if out_of_range_bbox(tracking_info[each_pre_index], width, height, offset):
                    count_out += 1
                    # if target_obj_id == 10:
                    #     print("count out first")
                else:
                    if validate_center(pre_obj_center, False, roi_list, height):
                        count_in += 1
                        # if target_obj_id == 10:
                        #     print("count in")
                    else:
                        count_out += 1
                        # if target_obj_id == 10:
                        #     print("count out second")
    return count_out, count_in, exist_latest_obj


def center_box(cur_box):
    return (int((cur_box[0]+cur_box[2])/2), int((cur_box[1]+cur_box[3])/2))



def voting(point, vote_movement, obj_id, moi_list):
    # one vote for each time point lying inside MOI
    exist_MOI = False
    if obj_id not in vote_movement:
        vote_movement[obj_id] = {}
    for moi_id in moi_list:
        moi_content = moi_list[moi_id]
        if moi_content[int(point[1])][int(point[0])] == True:
            if moi_id not in vote_movement[obj_id]:
                vote_movement[obj_id][moi_id] = 0
            vote_movement[obj_id][moi_id] += 1
            exist_MOI = True

    return vote_movement, exist_MOI

def add_num_class_count(num_class_out, class_name, max_movement_id):

    if max_movement_id not in num_class_out:
        num_class_out[max_movement_id] = 0

    num_class_out[max_movement_id] += 1
    num_object_out = num_class_out[max_movement_id]
    class_type = class_name
    
    return num_class_out, num_object_out, class_type

def car_counting(vid_name, roi_list, moi_list):
    print("Processing", vid_name)
    video_name = vid_name+".mp4"
    tracking_info = np.load(PATH_TRACKING + '/info_' + vid_name + '.npy', allow_pickle = True)
    N = tracking_info.shape[0]
    frame_id = tracking_info[:, 1].astype(np.int).reshape(N)
    delta_fix = 20
    obj_id = tracking_info[:, 3].astype(np.int).reshape(N)
    results = []

    input = cv2.VideoCapture(PATH_VIDEO + '/' + vid_name + '.mp4')
    width = int(input.get(3)) # get width
    height = int(input.get(4)) #get height

    num_c1_out = {} # each key is mv_id, each value count number of c1

    num_c2_out = {} # each key is mv_id, each value count number of c2

    num_c3_out = {} # each key is mv_id, each value count number of c3

    num_c4_out = {} # each key is mv_id, each value count number of c4

    vote_movement = {} # each key is obj_id, each value is A dictionary. Each key in A is movement_id, 
    #each value in A is count vote for that object to the corresponding 
    already_count = []


    for fr_id in range(1, max(frame_id)+1):
        index_cur_fr = np.where(frame_id==fr_id)[0]
        for index_box in index_cur_fr:
            cur_box = tracking_info[index_box][4:]
            cur_center = center_box(cur_box)
            cur_obj_id = tracking_info[index_box][3]

            is_inside_roi = validate_center(cur_center, False, roi_list, height)
            vote_movement, inside_MOI = voting(cur_center, vote_movement, cur_obj_id, moi_list)

            discrete_dict_class = {1: num_c1_out, 
                                    2: num_c2_out, 
                                    3: num_c3_out, 
                                    4: num_c4_out}
            if not inside_MOI:
                continue 
            
            if not is_inside_roi:# current car is outside roi
                count_out, count_in, is_ok = find_latest_object_and_vote_direction(frame_id, fr_id, tracking_info, 10, cur_obj_id, roi_list, width, height)
                # if cur_obj_id == 10:
                #     print(count_out, count_in)
                if is_ok: #exist object
                    # pre_obj_center = center_box(latest_obj[4:])
                    # is_inside_roi_pre_obj = validate_center(pre_obj_center, False, roi_list)
                    if count_in>=count_out and cur_obj_id not in already_count: # previous car lies inside roi
                        already_count.append(cur_obj_id)
                        max_movement_id = max(vote_movement[cur_obj_id], key=vote_movement[cur_obj_id].get)
                    
                        class_id = int(tracking_info[index_box][0])
                        num_class_out = discrete_dict_class[class_id]

                        num_class_out, num_object_out, class_type = add_num_class_count(num_class_out, 
                                                                    "class_"+str(class_id), 
                                                                    max_movement_id)

                        results.append([fr_id, num_object_out, cur_center[0], cur_center[1], max_movement_id, class_type])
            else : # using offset to refine again(outside video cam)
                is_out = out_of_range_bbox(tracking_info[index_box], width, height, 2)
                
                if is_out:
                    count_out, count_in, is_ok = find_latest_object_and_vote_direction(frame_id, fr_id, tracking_info, 10, cur_obj_id, roi_list, width, height)
                    # if cur_obj_id == 10:
                    #     print(count_out, count_in)
                    if is_ok: #exist object
                        # pre_obj_center = center_box(latest_obj[4:])
                        # is_inside_roi_pre_obj = validate_center(pre_obj_center, False, roi_list)
                        if count_in>=count_out and cur_obj_id not in already_count: # previous car lies inside roi
                            already_count.append(cur_obj_id)
                            max_movement_id = max(vote_movement[cur_obj_id], key=vote_movement[cur_obj_id].get)

                            class_id = int(tracking_info[index_box][0])
                            num_class_out = discrete_dict_class[class_id]

                            num_class_out, num_object_out, class_type = add_num_class_count(num_class_out, 
                                                                        "class_"+str(class_id), 
                                                                        max_movement_id)

                            results.append([fr_id, num_object_out, cur_center[0], cur_center[1], max_movement_id, class_type])
            if fr_id == 900:
                break

    if VISUALIZED:
        output = cv2.VideoWriter(PATH_SVIDEO + '/' + vid_name + '.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 5.0, (width, height))
        idx = 0
        results = np.array(results)
        N = len(results)
        print(N)
        frame_id = results[:, 0].astype(np.int).reshape(N)
        text_summary = build_text_name_dict(moi_list)

        while (input.isOpened()):
            ret, frame = input.read()
            if not ret:
                break
            idx += 1
            indx_cur_fr = np.where(frame_id == idx)[0]
            annotate_fr = draw_roi(roi_list, frame)
            if len(indx_cur_fr)!=0:
                for result_id in indx_cur_fr:
                    cur_annotate = results[result_id] 
                    count_object = cur_annotate[1]
                    cv2.putText(annotate_fr, cur_annotate[-1]+"-"+str(count_object).zfill(5), (int(cur_annotate[2]), int(cur_annotate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA) 
            annotate_fr = draw_roi(roi_list, frame)
            annotate_fr = draw_moi(annotate_fr, vid_name)
            #draw text results
            if len(indx_cur_fr)!=0:
                for result_id in indx_cur_fr:
                    cur_annotate = results[result_id] 
                    count_object = cur_annotate[1]
                    moi_id = cur_annotate[4]
                    class_type = cur_annotate[5]
                    text_summary[moi_id+"_"+class_type] = count_object
                    cv2.putText(annotate_fr, cur_annotate[4]+"-"+str(count_object).zfill(5), (int(cur_annotate[2]), int(cur_annotate[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA) 
            annotate_fr = draw_text_summarize(annotate_fr, text_summary, width, height)
            output.write(annotate_fr)
            if idx == 900:
                break
        input.release()
        output.release()

    # np.save(PATH_RESULTS + '/info_' + vid_name+".mp4", results)
    return results

if __name__ == "__main__":
    if not os.path.isdir(PATH_RESULTS):
        os.mkdir(PATH_RESULTS)

    if not os.path.isdir(PATH_SVIDEO):
        os.mkdir(PATH_SVIDEO)

    moi_list = load_moi()
    roi_list = load_roi()

    for video_name in os.listdir(PATH_VIDEO):
        # if video_name == "cam_18.mp4": #video_name .endswith(".mp4") or video_name == "cam_1.mp4":
        roi_vid_name = video_name[:-4].split("_")[0]+ "_" + video_name[:-4].split("_")[1]
        results = car_counting(video_name[:-4], roi_list[roi_vid_name], moi_list[roi_vid_name])
