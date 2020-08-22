import numpy as np
import json
import os
back_bone = ["Center_net", "dla_backbone", "resnet50_backbone", "dla_34_final"]
PATH_ID_LIST = "./list_video_id.txt"
PATH_COUNTING_RESULTS = '../../baseline_results/info_counting'
out_csv_file = "../../baseline_results/submissions.csv"
out_txt_file = "../../baseline_results/submissions_text.txt"
def build_mapping_dictionary():
    file_id = open(PATH_ID_LIST, "r")
    file_list = file_id.read().splitlines()
    my_dict = {}
    for each in file_list:
        each_id = each.split(" ")[0]
        each_filename = each.split(" ")[1][:-4]
        my_dict[each_filename] = each_id
    return my_dict

def write_submission():
    my_dict = {}
    class_type = {"class_1": 1, "class_2": 2, "class_3": 3, "class_4": 4}
    csv_file = open(out_csv_file, "w+")
    # csv_file.write(','.join(['video_clip_id', 'frame_id', 'movement_id', 'vehicle_class_id']))
    # csv_file.write('\n')
    count = 0
    for file_counting in os.listdir(PATH_COUNTING_RESULTS):
        print("Num file processing:", count)
        results_counting = np.load(os.path.join(PATH_COUNTING_RESULTS, file_counting))

        vid_name = file_counting[5:-8]
        print("Video name", vid_name)
        results_sort = sorted(results_counting, key = lambda x:int(x[0]))
        for each in results_sort:
            content = [vid_name, str(each[0]), str(each[4]), str(class_type[each[-1]])]
            csv_file.write(','.join(content))
            csv_file.write('\n')
        count += 1

    csv_file.close()

def write_submission_txt():
    class_type = {"class_1": 1, "class_2": 2, "class_3": 3, "class_4": 4}
    with open(out_txt_file, 'w') as result_file:
        for file_counting in os.listdir(PATH_COUNTING_RESULTS):
            results_counting = np.load(os.path.join(PATH_COUNTING_RESULTS, file_counting))

            vid_name = file_counting[5:-8]
            print("Video name", vid_name)
            results_sort = sorted(results_counting, key = lambda x:int(x[0]))
            for each in results_sort:
                result_file.write('{} {} {} {}\n'.format(vid_name, each[0], each[4], class_type[each[-1]]))

if __name__ == "__main__":
    # write_submission()
    # map_dict = build_mapping_dictionary()
    # write_submission()
    write_submission_txt()


