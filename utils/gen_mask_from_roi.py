import numpy as np 
import os 
import json
import cv2 

path_mask = "../../mask_roi"
PATH_ROI = "../../zones-movement_paths"

def gen_mask():
    if os.path.isdir(path_mask) == False:
        os.mkdir(path_mask)

    for annot_file in os.listdir(PATH_ROI):
        if annot_file.endswith(".json"):

            with open(os.path.join(PATH_ROI, annot_file)) as f_p:
                data = json.load(f_p)

            file_name = annot_file[:-5]
            polygon = data["shapes"]
            width = data["imageWidth"]
            height = data["imageHeight"]
            for each in polygon:
                if each["label"] == "zone":
                    empty_mask = np.zeros((height, width),dtype='int32') 

                    list_points = each["points"]
                    label = each["label"]
                    new_mask = cv2.fillConvexPoly(empty_mask, np.array(list_points, dtype="int32"), 255)
                    
                    cv2.imwrite(os.path.join(path_mask, file_name+".jpg"), new_mask)
                    np.save(os.path.join(path_mask, file_name), new_mask)

if __name__ == "__main__":
    gen_mask()