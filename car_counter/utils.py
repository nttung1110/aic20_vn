import numpy as np
import os
import matplotlib.path as mplPath
import cv2
import json

PATH_MOI = "../../MOIs/"
PATH_SCREENSHOT = "../../zones-movement_paths"
path_roi_mask = "../../mask_roi"

def load_movement_vector():
    mov_vec_dict = {}

    for file_name in os.listdir(PATH_SCREENSHOT):
        # From tail to head

        if file_name.endswith(".json"):
            mov_vec_dict[file_name[:-5]] = {}
            full_path_name = os.path.join(PATH_SCREENSHOT, file_name)

            with open(full_path_name) as json_file:
                all_shape = json.load(json_file)['shapes']

            for each in all_shape:
                if each["label"] != "zone":
                    label = each["label"]
                    list_points = each["points"]

                    x_tail, y_tail = list_points[0][0], list_points[0][1]
                    x_head, y_head = list_points[1][0], list_points[1][1]
                    mov_vec_dict[file_name[:-5]][label] = [(x_tail, y_tail), (x_head, y_head)]
    
    return mov_vec_dict


def load_roi():
    roi_list = {}
    for file_name in os.listdir(PATH_SCREENSHOT):
        if(file_name.endswith(".json")):
            full_path_name = os.path.join(PATH_SCREENSHOT, file_name)
            with open(full_path_name) as json_file:
                all_shape = json.load(json_file)['shapes']

            for each_shape in all_shape:
                if each_shape["label"] == "zone":
                    list_points = each_shape["points"]
                    break

            roi_list[file_name[:-5]] = list_points
    return roi_list

def load_roi_mask():
    roi_mask = {}

    for file_name in os.listdir(path_roi_mask):
        f_name = file_name[:-4]
        if file_name.endswith(".npy"):
            full_path_name = os.path.join(path_roi_mask, file_name)
            content = np.load(full_path_name)
            roi_mask[f_name] = content
    return roi_mask

def get_moi_name(vid_name):
    list_moi_name = []    
    full_path_name = os.path.join(PATH_SCREENSHOT, vid_name+".json")
    with open(full_path_name) as json_file:
        all_shape = json.load(json_file)['shapes']

    for each in all_shape:
        if each["label"] != "zone":
            list_moi_name.append(each["label"])

    return list_moi_name

def load_moi():
    moi_list = {}
    '''
        moi_list is a dictionary, each key is video name and its corresponding mask
        each value in each key is also a dictionary.This dictionary has key as movement_id in
        video and value is the the mask of the movement in binary 
    '''
    print("Extracting MOI")
    for folder_name in os.listdir(PATH_MOI):
        moi_list[folder_name] = {}
        for file_name in os.listdir(os.path.join(PATH_MOI, folder_name)):
            if file_name.endswith(".npy"):
                movement_id = file_name.split("_")[-1][:-4]
                full_path_name = os.path.join(PATH_MOI, folder_name, file_name)
                content = np.load(full_path_name)
                moi_list[folder_name][movement_id] = content
    return moi_list
    

def draw_roi(roi_list, image):
    start_point = (int(roi_list[0][0]), int(roi_list[0][1]))
    for end_point in roi_list[1:]:
        end_point = (int(end_point[0]), int(end_point[1]))
        cv2.line(image, start_point, end_point, (0,0,255), 2)
        start_point = end_point
    return image

def draw_moi_v1(annotated_frame, vid_name):
    with open(os.path.join(PATH_SCREENSHOT, vid_name+".json")) as json_file:
        content = json.load(json_file)['shapes']
        for element in content:
            if element["label"] != "zone":
                list_points = element['points']
                label = element['label']
                # draw arrow
                for index, point in enumerate(list_points):
                    start_point = (int(list_points[index][0]), int(list_points[index][1]))
                    end_point = (int(list_points[index+1][0]), int(list_points[index+1][1]))
                    if index+2 == len(list_points):
                        cv2.arrowedLine(annotated_frame, start_point, end_point, (255, 0, 0), 5)
                        break
                    else:
                        cv2.line(annotated_frame, start_point, end_point, (255,0,0), 5)
                cv2.putText(annotated_frame, label, end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA)
    return annotated_frame

def draw_moi(annotated_frame, vid_name):
    vid_name_json = vid_name.split("_")
    vid_name_json = vid_name_json[0]+"_"+vid_name_json[1]+".json"
    json_file = open(os.path.join(PATH_SCREENSHOT, vid_name_json))
    content = json.load(json_file)['shapes']
    for element in content:
        if element["label"] != "zone":
            list_points = element['points']
            label = element['label']
            # draw arrow
            for index, point in enumerate(list_points):
                start_point = (int(list_points[index][0]), int(list_points[index][1]))
                end_point = (int(list_points[index+1][0]), int(list_points[index+1][1]))
                if index+2 == len(list_points):
                    cv2.arrowedLine(annotated_frame, start_point, end_point, (255, 0, 0), 5)
                    break
                else:
                    cv2.line(annotated_frame, start_point, end_point, (255,0,0), 5)
            cv2.putText(annotated_frame, label, end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA)
    return annotated_frame


def draw_path(annotate_fr, offset_fr, cur_obj_id, tracking_info, frame_id_list, cur_fr_id):
    COLORS = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
    start_point = []
    for delta in range(0, offset_fr):
        pre_index = np.where(frame_id_list == (cur_fr_id - delta))[0]
        for each_pre_index in pre_index:
            if tracking_info[each_pre_index][3]==cur_obj_id:
                end_point = center_box(tracking_info[each_pre_index][4:])
                if len(start_point)!=0:
                    cv2.line(annotate_fr, start_point, end_point, COLORS, 10)
                start_point = end_point
    return annotate_fr

def draw_text_summarize(annotate_fr, text_name_dict, width, height):
    x_coor = 20
    y_coor = 20
    for text_name in text_name_dict:
        str_write = text_name+":"+str(text_name_dict[text_name])+"."
        cv2.putText(annotate_fr, str_write, (x_coor, y_coor), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA)
        x_coor += 200
        if x_coor + 200 >=width:
            x_coor = 20
            y_coor += 40

    return annotate_fr

def build_text_name_dict(moi_list):
    results = {}
    for moi_id in moi_list:
        text_name_c1 = moi_id+"_"+"class_1"
        text_name_c2 = moi_id+"_"+"class_2"
        text_name_c3 = moi_id+"_"+"class_3"
        text_name_c4 = moi_id+"_"+"class_4"

        results[text_name_c1] = 0
        results[text_name_c2] = 0
        results[text_name_c3] = 0
        results[text_name_c4] = 0

    results = dict(sorted (results.items()))
    return results

def build_text_name_dict_v1(moi_list):
    results = {}
    for moi_id in moi_list:
        results[str(moi_id)] = {}

        results[str(moi_id)]["class_1"] = 0
        results[str(moi_id)]["class_2"] = 0
        results[str(moi_id)]["class_3"] = 0
        results[str(moi_id)]["class_4"] = 0

    # results = dict(sorted (results.items()))
    return results

def draw_text_summarize_v1(annotate_fr, text_name_dict, width, height):
    x_coor = 20
    y_coor = 20
    write_text_dict = {}

    for moi_name in text_name_dict:
        write_text_dict[moi_name+"_"+"class_1"] = text_name_dict[moi_name]["class_1"]
        write_text_dict[moi_name+"_"+"class_2"] = text_name_dict[moi_name]["class_2"]
        write_text_dict[moi_name+"_"+"class_3"] = text_name_dict[moi_name]["class_3"]
        write_text_dict[moi_name+"_"+"class_4"] = text_name_dict[moi_name]["class_4"]

    for text_name in write_text_dict:
        str_write = text_name+":"+str(write_text_dict[text_name])+"."
        cv2.putText(annotate_fr, str_write, (x_coor, y_coor), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2,cv2.LINE_AA)
        x_coor += 300
        if x_coor + 300 >=width:
            x_coor = 20
            y_coor += 80

    return annotate_fr
    
