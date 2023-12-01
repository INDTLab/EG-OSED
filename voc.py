import sys

import xml.etree.ElementTree as ET
import os
from tqdm import tqdm


PROJECT_PATH=os.path.dirname(os.path.abspath(__file__))

def parse_voc_annotation(
    data_path, file_type, anno_path,use_difficult_bbox=False
):
    """
    phase pascal voc annotation, eg:[image_global_path xmin,ymin,xmax,ymax,cls_id]
    :param data_path: eg: VOC\VOCtrainval-2007\VOCdevkit\VOC2007
    :param file_type: eg: 'trainval''train''val'
    :param anno_path: path to ann file
    :param use_difficult_bbox: whither use different sample
    :return: batch size of data set
    """
    
    '''
    #fosd_classes:
    classes = ["Bay","Beach_Hut","Beach_Umbrella","Buoy","Cargo_Port","Coastal_Camping","Coastal_Nuclear_Power_Plant","Coastal_Road","Coastal_Windmill","Diving","Fleet","Headland","Island","Lighthouse","Offshore_Oil_Rig","Pier","Sailing","Sea_Arch","Sea_Bird","Sea_Bridge","Sea_Cave","Sea_Cliff","Sea_Farm","Sea_Iceberg","Sea_Mammal_Animal","Sea_Stack","Sea_Wall","Sea_Wave","Skerry","Star_Fish","Underwater_Fish","Underwater_Jellyfish","Underwater_Shark","Underwater_Turtle"]
    '''
  
    #places_classes
    classes=['aircraft_carrier', 'beach_hut', 'buoy', 'canoe', 'container_ship', 'crab', 'cruise', 'cruiser', 'diver', 'fish', 'fishing_boat', 'hippocampu', 'jellyfish', 'landing_ship', 'life_buoy', 'lighthouse', 'motorboat', 'ocean_rubbish', 'offshore_oilrig', 'oil_tanker', 'penguin', 'pier', 'rubber_boats', 'sailboat', 'sandcastle', 'seaplane', 'sea_bird', 'sea_bridge', 'sea_fork', 'sea_mammal', 'sea_turtle', 'shark', 'shrimp', 'sign', 'starfish', 'submarine', 'surfboard', 'tugboat', 'umbrella', 'yacht']
    
    
    '''
    #sun_classes
    classes = ["boathouse","bridge","iceberg","lighthouse","oilrig","sandbar"]
    '''

    ##########################################
    img_inds_file = os.path.join(
        data_path, "ImageSets", file_type + ".txt"
    )
    with open(img_inds_file, "r") as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    with open(anno_path, "a") as f:
        for image_id in tqdm(image_ids):
            new_str = ''
            ########################################
            image_path = os.path.join(
                data_path, "JPEGImages", image_id + ".jpg"
            )
            annotation = image_path
            label_path = os.path.join(
                data_path, "Annotations", image_id + ".xml"
            )
            root = ET.parse(label_path).getroot()
            objects = root.findall("object")
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                class_name= obj.find("name").text.strip()
                
                if class_name not in classes:
                    print(class_name)
                    continue
                if (not use_difficult_bbox) and (
                    int(difficult) == 1
                ):  # difficult 表示是否容易识别，0表示容易，1表示困难
                    continue
                bbox = obj.find("bndbox")
                class_id = classes.index(class_name)
                xmin = bbox.find("xmin").text.strip()
                ymin = bbox.find("ymin").text.strip()
                xmax = bbox.find("xmax").text.strip()
                ymax = bbox.find("ymax").text.strip()
                new_str += " " + ",".join(
                    [xmin, ymin, xmax, ymax, str(class_id)]
                )
            if new_str == '':
                continue
            annotation += new_str
            annotation += "\n"
            # print(annotation)
            f.write(annotation)
    return len(image_ids)


if __name__ == "__main__":

    #######################################
    #dataset dir
    train_data_path = os.path.join(
        PROJECT_PATH, "places365_OD"
    )
    val_data_path = os.path.join(
        PROJECT_PATH, "places365_OD"
    )

 
    train_annotation_path = os.path.join(PROJECT_PATH,"data/places365_OD", "train.txt")
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)   
    val_annotation_path = os.path.join(PROJECT_PATH,"data/places365_OD", "val.txt")
    if os.path.exists(val_annotation_path):
        os.remove(val_annotation_path)

    len_train = parse_voc_annotation(
        train_data_path,
        "train",
        train_annotation_path,
        use_difficult_bbox=False,
    )

    len_val = parse_voc_annotation(
        val_data_path,
        "val",
        val_annotation_path,
        use_difficult_bbox=False,
    )

    print(
        "The number of images for train and test are :train : {0} |  val : {1}".format(
            len_train, len_val
        )
     )
