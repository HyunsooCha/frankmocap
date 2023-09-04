import face_alignment
import os, glob, json
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import natsort
import time
import torch
import shutil
import warnings
warnings.filterwarnings("ignore")



def expand_bbox_asymmetrically_square(bbox, img_width, img_height, upper_height_factor=1.2, lower_height_factor=1.5):
    center_x, center_y, width, height = bbox

    new_upper_height = (height / 2) * upper_height_factor
    new_lower_height = (height / 2) * lower_height_factor

    new_height = new_upper_height + new_lower_height
    new_width = new_height

    new_xmin = center_x - new_width / 2
    new_ymin = center_y - new_upper_height
    new_xmax = center_x + new_width / 2
    new_ymax = center_y + new_lower_height

    # Clip the coordinates to the image bounds
    new_xmin = max(0, new_xmin)
    new_ymin = max(0, new_ymin)
    new_xmax = min(img_width, new_xmax)
    new_ymax = min(img_height, new_ymax)

    # Adjust coordinates to maintain square shape
    clipped_width = new_xmax - new_xmin
    clipped_height = new_ymax - new_ymin
    if clipped_width != clipped_height:
        delta = abs(clipped_width - clipped_height)

        if clipped_width > clipped_height:
            if new_ymax + delta <= img_height:
                new_ymax += delta
            else:
                new_ymin -= delta
        else:
            if new_xmax + delta <= img_width:
                new_xmax += delta
            else:
                new_xmin -= delta
    # print(new_xmax-new_xmin, new_ymax-new_ymin)
    return (new_xmin, new_ymin, new_xmax, new_ymax)



def main(args):
    root_path = args.root_path
    video_file = args.video_file
    output_file_size = tuple([args.output_file_size, args.output_file_size])
    total_frame = 4500 # 25 frames \times 60 seconds \times 3 minutes

    video_name = video_file.split('.')[0]
    raw_path = os.path.join(root_path, video_file)
    if not os.path.exists('mocap_output'):
        os.mkdir('mocap_output')
    path1 = os.path.join('mocap_output', video_name+'_frames')
    path2 = os.path.join('mocap_output', video_name+'_cropped_frames')
    output_dir = os.path.join('mocap_output', video_name+'_mocap_output')
    if os.path.exists(path1):
        shutil.rmtree(path1)
    if os.path.exists(path2):
        shutil.rmtree(path2)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.system('ffmpeg -i {} '.format(raw_path)+' -qscale:v 2 -vsync 0 ' +os.path.join(path1, '%07d.png')) # -vf "fps={}" 
    
    print('[INFO] save hand bounding boxes using frankmocap...')
    start = time.time()
    # NOTE previous version
    # os.system('./xvfb-run-safe python -m demo.demo_handmocap --input_path {} --out_dir {} --view_type ego_centric --save_bbox_output'.format(path1, output_dir))
    os.system('./run_frankmocap.sh {} {}'.format(path1, output_dir))
    # os.system('xvfb-run -a python -m demo.demo_handmocap --input_path {} --out_dir {} --view_type ego_centric --save_bbox_output'.format(path1, output_dir))
    # os.system('python -m demo.demo_handmocap --input_path {} --out_dir {} --view_type ego_centric --save_bbox_output --renderer_type pytorch3d'.format(path1, output_dir))
    print('[INFO] save hand bounding boxes complete! time: ', time.time()-start)

    print('[INFO] get the face bounding box and predict the face keypoint using face alignment...')
    start = time.time()
    
    face_keypoints_filename = '{}_face_keypoints_{}.npy'.format(video_name, total_frame)
    if not os.path.exists(os.path.join('mocap_output', face_keypoints_filename)):
        face_detector = 'sfd'
        face_detector_kwargs = {
            'filter_threshold': 0.99,
        }
        try:
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector=face_detector, face_detector_kwargs=face_detector_kwargs, device='cuda')
        except:
            fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, face_detector=face_detector, face_detector_kwargs=face_detector_kwargs, device='cuda')

        preds = fa.get_landmarks_from_directory(path1)
        print('[INFO] prediction by face alignment complete! time: ', time.time()-start)

        np.save(os.path.join('mocap_output', face_keypoints_filename), preds)
        fa.device = 'cpu' # to save cuda memory...
    else:
        preds = np.load(os.path.join('mocap_output', face_keypoints_filename))
        print('[INFO] load face keypoints complete! time: ', time.time()-start)
    idx = natsort.natsorted(preds)
    # half_size = output_file_size[0]//2
    

    print('[INFO] cropping...')

    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    start = time.time()
    order = 0

    # height = 0
    init_check = True

    for i, file in enumerate(sorted(glob.glob(os.path.join(output_dir, 'bbox', '*.json')))):
        
        with open(file) as f:
            data = json.load(f)

        if preds[idx[i]] is None: # 얼굴이 없을 경우
            continue
        if len(preds[idx[i]]) != 1: # 얼굴이 2개 이상일 경우
            continue
        if len(data['hand_bbox_list']) > 1: # 양손이 2개 이상일 경우 (예를 들어 다른 사람의 손)
            continue
        
        left_bbox = data['hand_bbox_list'][-1]['left_hand']
        right_bbox = data['hand_bbox_list'][-1]['right_hand']

        x1_face = np.min(preds[idx[i]][-1][:, 0]) # xmin
        y1_face = np.min(preds[idx[i]][-1][:, 1]) # ymin
        x2_face = np.max(preds[idx[i]][-1][:, 0]) # xmax
        y2_face = np.max(preds[idx[i]][-1][:, 1]) # ymax
        face_box = [x1_face, y1_face, x2_face-x1_face, y2_face-y1_face]


        center_x = np.mean(preds[idx[i]][-1][:, 0]) # face_box[0] + face_box[2]//2
        center_y = np.mean(preds[idx[i]][-1][:, 1]) # face_box[1] + face_box[3]//2

        img = Image.open(idx[i])

        if init_check: # to fix the height
            width = x2_face-x1_face
            height = y2_face-y1_face
            init_check = False

        # NOTE previously, 2.6, 2.6
        expanded_face_bbox = expand_bbox_asymmetrically_square((center_x, center_y, width, height), img_width=img.size[0], img_height=img.size[1], upper_height_factor=2.8, lower_height_factor=2.7)
        face_box = expanded_face_bbox

        if left_bbox is not None:
            hand_box = left_bbox
            # Compute the (x, y) coordinates of the intersection
            x1, y1 = max(face_box[0], hand_box[0]), max(face_box[1], hand_box[1])
            x2, y2 = min(face_box[0] + face_box[2], hand_box[0] + hand_box[2]), min(face_box[1] + face_box[3], hand_box[1] + hand_box[3])

            # Compute the area of the intersection
            intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

            # Compute the areas of the two bounding boxes
            face_area = face_box[2] * face_box[3]
            hand_area = hand_box[2] * hand_box[3]

            # Compute the overlap ratio
            overlap_ratio = intersection_area / min(face_area, hand_area)

            # Print the overlap ratio
            # print("Overlap ratio:", overlap_ratio)
            if overlap_ratio > 0.0:
                continue

        if right_bbox is not None:
            hand_box = right_bbox
            # Compute the (x, y) coordinates of the intersection
            x1, y1 = max(face_box[0], hand_box[0]), max(face_box[1], hand_box[1])
            x2, y2 = min(face_box[0] + face_box[2], hand_box[0] + hand_box[2]), min(face_box[1] + face_box[3], hand_box[1] + hand_box[3])

            # Compute the area of the intersection
            intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

            # Compute the areas of the two bounding boxes
            face_area = face_box[2] * face_box[3]
            hand_area = hand_box[2] * hand_box[3]

            # Compute the overlap ratio
            overlap_ratio = intersection_area / min(face_area, hand_area)

            # Print the overlap ratio
            # print("Overlap ratio:", overlap_ratio)
            if overlap_ratio > 0.0:
                continue        

        results = yolo_model(img)
        class_name_dict = {index: name for index, name in results.names.items()}
        # Filter the detected objects by class label 'cup' and a confidence threshold

        # occlusion_candidate = [
        #     'cup', 'baseball glove', 'tennis racket', 'bottle', 'wine glass', 
        #     'fork', 'knife', 'spoon', 'bowl', 'banana', 
        #     'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
        #     'hot dog', 'pizza', 'donut', 'cake', 'cell phone', 
        #     'scissors', 'hair drier', 'toothbrush'
        #     ]
        if False:
            occlusion_candidate = [
                'cup', 'bottle', 'wine glass', 
                'fork', 'knife', 'spoon', 'bowl', 'cell phone', 'scissors', 
                ]
            for label in occlusion_candidate:
                # cup_class_index = [k for k, v in class_name_dict.items() if v == 'cup'][0]
                cup_class_index = [k for k, v in class_name_dict.items() if v == label][0]
                cup_results = results.xyxy[0][results.xyxy[0][:, 5] == cup_class_index]
                cup_results = cup_results[cup_results[:, 4] > 0.1]

                if len(cup_results) > 0:
                    print('[INFO] {} detected!'.format(label))
                    for cups in range(cup_results.shape[0]):
                        cup_xmin, cup_ymin, cup_xmax, cup_ymax, cup_confidence, cup_class = cup_results[cups]
                        cup_xmin, cup_ymin, cup_xmax, cup_ymax = cup_xmin.item(), cup_ymin.item(), cup_xmax.item(), cup_ymax.item()
                        hand_box = [cup_xmin, cup_ymin, cup_xmax-cup_xmin, cup_ymax-cup_ymin]
                        # Compute the (x, y) coordinates of the intersection
                        x1, y1 = max(face_box[0], hand_box[0]), max(face_box[1], hand_box[1])
                        x2, y2 = min(face_box[0] + face_box[2], hand_box[0] + hand_box[2]), min(face_box[1] + face_box[3], hand_box[1] + hand_box[3])

                        # Compute the area of the intersection
                        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

                        # Compute the areas of the two bounding boxes
                        face_area = face_box[2] * face_box[3]
                        hand_area = hand_box[2] * hand_box[3]

                        # Compute the overlap ratio
                        overlap_ratio = intersection_area / min(face_area, hand_area)

                        # Print the overlap ratio
                        # print("Overlap ratio:", overlap_ratio)
                        if overlap_ratio > 0.0:
                            continue


        area = expanded_face_bbox

        cropped_img = img.crop(area)

        cropped_img = cropped_img.resize(output_file_size, Image.LANCZOS)

        file_name = str(order).zfill(7)+'.png' # i.split('/')[-1]
        order += 1
        cropped_img.save(os.path.join(path2, file_name))
        if cropped_img.size != output_file_size:
            print('Size Error')
        if order > total_frame:
            break

    print('[INFO] cropping complete! time: ', time.time()-start)
    # os.system('ffmpeg -framerate 25 -i ' + os.path.join(path2, '%07d.png') + ' -c:v libx264 -profile:v high422 -pix_fmt yuv420p -c:a copy '+os.path.join(root_path, video_name.replace('_temp', '')+'.mp4'))
    os.system('ffmpeg -y -framerate 25 -i ' + os.path.join(path2, '%07d.png') + ' -c:v libx264 -profile:v high422 -pix_fmt yuv420p -c:a copy '+os.path.join(root_path, video_name.replace('_orig', '')+'.mp4'))
    
    print('[INFO] video saved! time: ', time.time()-start)
    return path1, path2, output_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/root/GitHub/frankmocap/sample_data/')
    parser.add_argument('--video_file', type=str, default='friendshipping_1.mp4')
    parser.add_argument('--output_file_size', type=int, default=512)

    args = parser.parse_args()

    path1, path2, output_dir = main(args)


