import os
from nuscenes.nuscenes import NuScenes
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
import torch

def build_transformation(translation, rotation):
    q = Quaternion(rotation)    # create a Quaternion object from [w, x, y, z]
    R = q.rotation_matrix   # 3x3 rotaton matrixi
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T

def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


class NuscenesDataset(Dataset):
    def __init__(self, dataroot, version='v1.0-trainval', transform=None):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        
        self.transform = transform
        good_tokens = []
        missing = 0

        for sample in self.nusc.sample:
            cam_token = sample['data']['CAM_FRONT']
            cam_data  = self.nusc.get('sample_data', cam_token)
            img_path  = os.path.join(self.nusc.dataroot, cam_data['filename'])

            if os.path.exists(img_path):
                good_tokens.append(cam_token)
            else:
                missing += 1

        if missing:
            print(f"Skipping {missing} samples because their images aren't unzipped.")

        self.sample_data_tokens = good_tokens

    def __len__(self):
        return len(self.sample_data_tokens)

    def __getitem__(self, idx):
        token = self.sample_data_tokens[idx]
        cam_data = self.nusc.get('sample_data', token)
        
        # Get calibrated sensor data for the front camera
        cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        
        # Load the front camera intrinsic matrix
        intrinsic = None
        if 'camera_intrinsic' in cam_calib:
            intrinsic = np.array(cam_calib['camera_intrinsic'])
        
        image_path = os.path.join(self.nusc.dataroot, cam_data['filename'])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        sample = self.nusc.get('sample', cam_data['sample_token'])
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        pcd_path = os.path.join(self.nusc.dataroot, lidar_data['filename'])
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(f"Point cloud file not found: {pcd_path}")
        pcd = LidarPointCloud.from_file(pcd_path).points.T 

        # Apply a pcd filter 
        valid_ind = (pcd[:, 0] < -3) | (pcd[:, 0] > 3) | (pcd[:, 1] < -3) | (pcd[:, 1] > 3)
        pcd = pcd[valid_ind, :]

        # lidar2ego
        lidar_calib = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        r_lidar2ego = torch.Tensor(Quaternion(lidar_calib['rotation']).rotation_matrix)
        t_lidar2ego = torch.Tensor(lidar_calib['translation'])
        T_lidar2ego = torch.eye(4)
        T_lidar2ego[:3,:3] = r_lidar2ego
        T_lidar2ego[:3, 3] = t_lidar2ego

        # camera2ego
        r_cam2ego  = torch.Tensor(Quaternion(cam_calib['rotation']).rotation_matrix)
        t_cam2ego  = torch.Tensor(cam_calib['translation'])
        T_cam2ego  = torch.eye(4)
        T_cam2ego[:3,:3] = r_cam2ego
        T_cam2ego[:3, 3] = t_cam2ego

        # lidar2camera
        T_ego2cam = torch.inverse(T_cam2ego)
        gt_transform = T_ego2cam @ T_lidar2ego

        return image, pcd, gt_transform, intrinsic


    def sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

def img_transform(img, post_rot, post_tran,
                resize, resize_dims, crop,
                flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)
    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b
    return img, post_rot, post_tran



if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataroot = '/data/HangQiu/data/nuscenes'
    dataset = NuscenesDataset(dataroot=dataroot, version='v1.0-trainval', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(len(dataset))
    for batch_idx, (images, pcds, gt_transforms, intrinsics) in enumerate(dataloader):
        # print(f"Batch {batch_idx}:")
        # print("  Images shape:", images.shape)  # e.g., [4, 3, 224, 224]
        # print("  Example point cloud shape:", pcds[0].shape)  
        # print("  GT transform sample:\n", gt_transforms[0] if isinstance(gt_transforms, list) else gt_transforms)
        # print("  Intrinsics sample:\n", intrinsics[0] if isinstance(intrinsics, list) else intrinsics)
        break

