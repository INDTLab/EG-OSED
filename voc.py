import sys

import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import shutil


PROJECT_PATH=os.path.dirname(os.path.abspath(__file__))

def parse_voc_annotation(
    dataset,data_path, file_type, anno_path,use_difficult_bbox=False
):

    #fosd_classes:
    if dataset == 'FOSD_OD':
        classes = ["Bay","Beach_Hut","Beach_Umbrella","Buoy","Cargo_Port","Coastal_Camping","Coastal_Nuclear_Power_Plant","Coastal_Road","Coastal_Windmill","Diving","Fleet","Headland","Island","Lighthouse","Offshore_Oil_Rig","Pier","Sailing","Sea_Arch","Sea_Bird","Sea_Bridge","Sea_Cave","Sea_Cliff","Sea_Farm","Sea_Iceberg","Sea_Mammal_Animal","Sea_Stack","Sea_Wall","Sea_Wave","Skerry","Star_Fish","Underwater_Fish","Underwater_Jellyfish","Underwater_Shark","Underwater_Turtle"]
    #places_classes
    elif dataset  == 'Places365_OD':
        classes=['aircraft_carrier', 'beach_hut', 'buoy', 'canoe', 'container_ship', 'crab', 'cruise', 'cruiser', 'diver', 'fish', 'fishing_boat', 'hippocampu', 'jellyfish', 'landing_ship', 'life_buoy', 'lighthouse', 'motorboat', 'ocean_rubbish', 'offshore_oilrig', 'oil_tanker', 'penguin', 'pier', 'rubber_boats', 'sailboat', 'sandcastle', 'seaplane', 'sea_bird', 'sea_bridge', 'sea_fork', 'sea_mammal', 'sea_turtle', 'shark', 'shrimp', 'sign', 'starfish', 'submarine', 'surfboard', 'tugboat', 'umbrella', 'yacht']
     #sun_classes
    elif dataset  == 'SUN_OD':
        classes = ["boathouse","bridge","iceberg","lighthouse","oilrig","sandbar"]
 
    img_inds_file = os.path.join(
        data_path, "ImageSets", file_type + ".txt"
    )
    with open(img_inds_file, "r") as f:
        lines = f.readlines()
        image_ids = [line.strip() for line in lines]

    with open(os.path.join(anno_path,file_type + ".txt"), "a") as f:
        for image_id in tqdm(image_ids):
            new_str = ''
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

    for dataset in ["FOSD_OD","Places365_OD","SUN_OD"]:
        train_data_path = os.path.join(
            PROJECT_PATH, 'dataset',dataset
        )
        test_data_path = os.path.join(
            PROJECT_PATH, 'dataset',dataset
        )

     
        train_annotation_path = os.path.join(PROJECT_PATH,"data",dataset)
        if os.path.exists(train_annotation_path):
            shutil.rmtree(train_annotation_path)
        os.makedirs(train_annotation_path)
        test_annotation_path = os.path.join(PROJECT_PATH,"data",dataset)
        if os.path.exists(test_annotation_path):
            shutil.rmtree(test_annotation_path)
        os.makedirs(test_annotation_path)

        len_train = parse_voc_annotation(
            dataset,
            train_data_path,
            "train",
            train_annotation_path,
            use_difficult_bbox=False,
        )

        len_test = parse_voc_annotation(
            dataset,
            test_data_path,
            "test",
            test_annotation_path,
            use_difficult_bbox=False,
        )

        print(
            "The number of images of {0}for train and test are :train : {1} |  test : {2}".format(
                dataset,len_train, len_test
            )
         )

