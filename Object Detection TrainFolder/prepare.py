import os
import shutil
import xml.etree.ElementTree as ET
import yaml

# 1. Define paths
workspace_images = r'preLabeling\workspace\images'
dataset_dir = r'dataset'

classes = ['axe', 'pickaxe', 'shovel']

# 2. Create YOLO structure
for split in ['train', 'valid']:
    os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)

def convert_and_move(source_folder, target_split):
    image_out = os.path.join(dataset_dir, target_split, 'images')
    label_out = os.path.join(dataset_dir, target_split, 'labels')
    
    for filename in os.listdir(source_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # Move image
            shutil.copy(os.path.join(source_folder, filename), os.path.join(image_out, filename))
            
            # Convert XML
            xml_name = filename.rsplit('.', 1)[0] + '.xml'
            xml_path = os.path.join(source_folder, xml_name)
            
            if os.path.exists(xml_path):
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                w, h = int(size.find('width').text), int(size.find('height').text)
                
                with open(os.path.join(label_out, filename.rsplit('.', 1)[0] + '.txt'), 'w') as f:
                    for obj in root.findall('object'):
                        cls_name = obj.find('name').text
                        if cls_name not in classes: continue
                        cls_id = classes.index(cls_name)
                        box = obj.find('bndbox')
                        xmin, xmax = float(box.find('xmin').text), float(box.find('xmax').text)
                        ymin, ymax = float(box.find('ymin').text), float(box.find('ymax').text)
                        
                        # YOLO conversion logic
                        x_center = ((xmin + xmax) / 2) / w
                        y_center = ((ymin + ymax) / 2) / h
                        width = (xmax - xmin) / w
                        height = (ymax - ymin) / h
                        f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

name_map = {i: name for i, name in enumerate(classes)}

# 3. Using it in your detection loop
def process_results(results):
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            
            # Access the name via the list index
            detected_name = classes[class_id]
            
            print(f"Detected: {detected_name}")

data_config = {
    'path': dataset_dir,
    'train': 'train/images',
    'val': 'valid/images',
    'names': name_map
}

with open(dataset_dir + '\data.yaml', 'w') as file:
    yaml.dump(data_config, file, default_flow_style=False, sort_keys=False)

# 4. Run conversion for your existing folders
convert_and_move(os.path.join(workspace_images, 'train'), 'train')
convert_and_move(os.path.join(workspace_images, 'test'), 'valid')

print("Dataset prepared successfully in the 'dataset' folder!")