import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob
from tqdm import tqdm
import argparse

# LLVIP数据集只有一个类别 'person'
classes = ["person"]

def convert(size, box):
    # 将VOC格式的 (xmin, ymin, xmax, ymax) 转换为YOLO格式的 (center_x, center_y, width, height) 归一化坐标
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(annotation_path, image_path, txt_save_path):
    # 确保保存路径存在
    os.makedirs(txt_save_path, exist_ok=True)
    
    # 获取所有图片文件的基础名（不带后缀）
    image_filenames = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(image_path, '*.jpg'))}
    
    print(f"在 {image_path} 中找到了 {len(image_filenames)} 张图片。")
    print(f"开始在 {annotation_path} 中查找并转换匹配的XML文件...")

    xml_files = glob.glob(os.path.join(annotation_path, '*.xml'))
    
    for xml_file in tqdm(xml_files, desc="Converting annotations"):
        image_id = os.path.splitext(os.path.basename(xml_file))[0]
        
        # 只有当图片存在时才进行转换
        if image_id in image_filenames:
            out_file = open(os.path.join(txt_save_path, f'{image_id}.txt'), 'w')
            
            tree=ET.parse(xml_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult)==1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert((w,h), b)
                out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            out_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to the folder with XML annotations')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the folder with corresponding images')
    parser.add_argument('--txt_save_path', type=str, required=True, help='Path to the folder where TXT files will be saved')
    opt = parser.parse_args()
    
    convert_annotation(opt.annotation_path, opt.image_path, opt.txt_save_path)
    print("\n转换完成！")