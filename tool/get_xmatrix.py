import glob
import json
import math
import operator
import os
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from tool.utils import *

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


###########################################################
def get_matrix(MINOVERLAP, path = './map_out',save_path=None, dataset=None):
    GT_PATH             = os.path.join(path, 'ground-truth')
    DR_PATH             = os.path.join(path, 'detection-results')
    TEMP_FILES_PATH     = os.path.join(path, '.temp_files')
    gt_classes=load_class_names('data/'+dataset+'.names')

    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)

    n_classes   = len(gt_classes)
    
    class_dict={}
    for i,j in enumerate(gt_classes):
        class_dict[j]=i


    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()

    for txt_file in ground_truth_files_list:
        file_id     = txt_file.split(".txt", 1)[0]
        file_id     = os.path.basename(os.path.normpath(file_id))
        
        lines_list      = file_lines_to_list(txt_file)
        bounding_boxes  = []
        is_difficult    = False
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except:
                if "difficult" in line:
                    line_split  = line.split()
                    _difficult  = line_split[-1]
                    bottom      = line_split[-2]
                    right       = line_split[-3]
                    top         = line_split[-4]
                    left        = line_split[-5]
                    class_name  = ""
                    for name in line_split[:-5]:
                        class_name += name + " "
                    class_name  = class_name[:-1]
                    is_difficult = True
                else:
                    line_split  = line.split()
                    bottom      = line_split[-1]
                    right       = line_split[-2]
                    top         = line_split[-3]
                    left        = line_split[-4]
                    class_name  = ""
                    for name in line_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]

            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})

        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except:
                    line_split      = line.split()
                    bottom          = line_split[-1]
                    right           = line_split[-2]
                    top             = line_split[-3]
                    left            = line_split[-4]
                    confidence      = line_split[-5]
                    tmp_class_name  = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name  = tmp_class_name[:-1]

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})

        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)


    #confusion matrix
    confusion_matrix = np.zeros(shape=[n_classes + 1, n_classes + 1])
    

    #遍历每个类
    for class_index, class_name in enumerate(gt_classes):
        
        dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
        dr_data = json.load(open(dr_file))
        nd          = len(dr_data)
        score       = [0] * nd
        score05_idx = 0
        #遍历每个类的每个框
        for idx, detection in enumerate(dr_data):
            dt_match=0#判断是否有真实框与预测框匹配
            file_id     = detection["file_id"]
            score[idx]  = float(detection["confidence"])
            
            if score[idx] > 0.5:
                gt_file             = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data   = json.load(open(gt_file))
                ovmax       = -1
                gt_match    = -1
                bb          = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:#选出与预测框iou最大的真实框
                    bbgt    = [ float(x) for x in obj["bbox"].split() ]
                    bi      = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw      = bi[2] - bi[0] + 1
                    ih      = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

                min_overlap = MINOVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            dt_match+=1
                            if gt_match["class_name"]==class_name:                                
                                gt_match["used"] = True
                                with open(gt_file, 'w') as f:
                                        f.write(json.dumps(ground_truth_data))
                            confusion_matrix [class_dict[gt_match["class_name"]],class_dict[class_name]]+=1
            if dt_match==0:
                confusion_matrix[-1, class_dict[class_name]] += 1
               

        
    for gt_file in glob.glob(TEMP_FILES_PATH+'/*_ground_truth.json') :
            ground_truth_data=json.load(open(gt_file))
            for obj in ground_truth_data:
                if not bool(obj["used"]):
                    confusion_matrix[class_dict[obj["class_name"]],-1]+=1
    plot_confusion_matrix(confusion_matrix, gt_classes + ['background'], save_path)
    shutil.rmtree(TEMP_FILES_PATH)#移除


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_path=None,
                          show=False,
                          title='Normalized Confusion Matrix',
                          color_theme='plasma'):
    """Draw confusion matrix with matplotlib.
    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
    """
    # normalize the confusion matrix
    per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]#增加维度-->(35,1)
    confusion_matrix = confusion_matrix.astype(np.float32) / per_label_sums*100 
    

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)#添加渐变条,mappable映射颜色的对象，ax在哪里显示

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                '{:.2f}'.format(
                    confusion_matrix[
                        i,
                        j] if not np.isnan(confusion_matrix[i, j]) else -1),
                ha='center',
                va='center',
                color='w',
                size=4)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1
    ###############################################################
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path+'.png', format='png')
    if show:
        plt.show()

   #####################################################     
if __name__=="__main__":
    MINOVERLAP=0.5
    map_out_path='map_out/sun/res50/p7'
    get_matrix(MINOVERLAP,map_out_path)

