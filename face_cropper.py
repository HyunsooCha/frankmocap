import face_alignment
import os, glob, json
from tqdm import tqdm
from PIL import Image
import numpy as np
import argparse
import natsort
import time

def main(args):
    root_path = args.root_path
    video_file = args.video_file
    output_file_size = tuple([args.output_file_size, args.output_file_size])

    video_name = video_file.split('.')[0]
    raw_path = os.path.join(root_path, video_file)
    path1 = os.path.join('mocap_output', video_name+'_frames')
    path2 = os.path.join('mocap_output', video_name+'_cropped_frames')
    output_dir = os.path.join('mocap_output', video_name+'_mocap_output')
    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.system('ffmpeg -i {} '.format(raw_path)+' -qscale:v 2 -vsync 0 ' +os.path.join(path1, '%07d.png')) # -vf "fps={}" 

    print('[INFO] prediction by face alignment...')
    start = time.time()
    face_detector = 'sfd'
    face_detector_kwargs = {
        'filter_threshold': 0.99,
    }
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector=face_detector, face_detector_kwargs=face_detector_kwargs, device='cuda')

    preds = fa.get_landmarks_from_directory(path1)
    print('[INFO] prediction by face alignment complete! time: ', time.time()-start)

    print('[INFO] sorting...')
    start = time.time()
    idx = natsort.natsorted(preds)
    # half_size = output_file_size[0]//2
    print('[INFO] sorting complete! time: ', time.time()-start)

    print('[INFO] save hand bounding boxes using frankmocap...')
    start = time.time()
    os.system('xvfb-run -a python -m demo.demo_handmocap --input_path {} --out_dir {} --view_type ego_centric --save_bbox_output'.format(path1, output_dir))
    # os.system('python -m demo.demo_handmocap --input_path {} --out_dir {} --view_type ego_centric --save_bbox_output --renderer_type pytorch3d'.format(path1, output_dir))
    print('[INFO] save hand bounding boxes complete! time: ', time.time()-start)

    print('[INFO] cropping...')
    start = time.time()
    order = 0
    for i, file in enumerate(sorted(glob.glob(os.path.join(output_dir, 'bbox', '*.json')))):
        if preds[idx[i]] is None: # 얼굴이 없을 경우
            continue
        if len(preds[idx[i]]) != 1: # 얼굴이 2개 이상일 경우
            continue
        
        with open(file) as f:
            data = json.load(f)

        if len(data['hand_bbox_list']) > 1: # 손이 2개 이상일 경우
            continue

        left_bbox = data['hand_bbox_list'][-1]['left_hand']
        right_bbox = data['hand_bbox_list'][-1]['right_hand']

        x1_face = np.min(preds[idx[i]][-1][:, 0])
        y1_face = np.min(preds[idx[i]][-1][:, 1])
        x2_face = np.max(preds[idx[i]][-1][:, 0])
        y2_face = np.max(preds[idx[i]][-1][:, 1])
        face_box = [x1_face, y1_face, x2_face-x1_face, y2_face-y1_face]

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
        
        img = Image.open(idx[i])

        center_x = np.mean(preds[idx[i]][-1][:, 0]) # face_box[0] + face_box[2]//2
        center_y = np.mean(preds[idx[i]][-1][:, 1]) # face_box[1] + face_box[3]//2

        # Compute the new (x, y) coordinates of the top-left corner
        new_x = max(0, center_x - output_file_size[0] // 2)
        new_y = max(0, center_y - output_file_size[1] // 2)

        # Ensure that the new (x, y) coordinates are inside the original image bounds
        if new_x + output_file_size[0] > img.size[0]:
            new_x = img.size[0] - output_file_size[0]
        if new_y + output_file_size[1] > img.size[1]:
            new_y = img.size[1] - output_file_size[1]

   
        area = (new_x, new_y, new_x+output_file_size[0], new_y+output_file_size[1])
        cropped_img = img.crop(area)
        file_name = str(order).zfill(7)+'.png' # i.split('/')[-1]
        order += 1
        cropped_img.save(os.path.join(path2, file_name))
        if cropped_img.size != output_file_size:
            print('Size Error')

    print('[INFO] cropping complete! time: ', time.time()-start)
    os.system('ffmpeg -framerate 25 -i ' + os.path.join(path2, '%07d.png') + ' -c:v libx264 -profile:v high422 -pix_fmt yuv420p -c:a copy '+os.path.join(root_path, video_name+'_cropped.mp4')) # 
    os.system('rm -rf '+path1)
    os.system('rm -rf '+path2)
    os.system('rm -rf '+output_dir)
    print('[INFO] video saved! time: ', time.time()-start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/stephencha/GitHub/frankmocap/sample_data/')
    parser.add_argument('--video_file', type=str, default='syuka_train_2.mp4')
    parser.add_argument('--output_file_size', type=int, default=512)

    args = parser.parse_args()

    main(args)




