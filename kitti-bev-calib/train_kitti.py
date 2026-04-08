import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from kitti_dataset import KittiDataset
from core.bev_calib import BEVCalib
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from torch.utils.data import random_split
import numpy as np
from pathlib import Path
from core.tools import generate_single_perturbation_from_T
import shutil
import cv2
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--dataset_root", type=str, default="YOUR_PATH_TO_KITTI/kitti-odemetry")
    parser.add_argument("--log_dir", type=str, default="./logs/kitti_default")
    parser.add_argument("--save_ckpt_per_epoches", type=int, default=-1)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--angle_range_deg", type=float, default=None)
    parser.add_argument("--trans_range", type=float, default=None)
    parser.add_argument("--eval_angle_range_deg", type=float, default=None)
    parser.add_argument("--eval_trans_range", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--eval_epoches", type=int, default=4)
    parser.add_argument("--deformable", type=int, default=-1)
    parser.add_argument("--bev_encoder", type=int, default=1)
    parser.add_argument("--xyz_only", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--scheduler", type=int, default=-1)
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    return parser.parse_args()

def crop_and_resize(item, size, intrinsics, crop=True):
    img = cv2.cvtColor(np.array(item), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if crop:
        mid_width = w // 2
        start_x = (w - mid_width) // 2
        cropped = img[:, start_x:start_x + mid_width]
        resized = cv2.resize(cropped, size)
    else:
        resized = cv2.resize(img, size)

    if crop:
        new_cx = intrinsics[0, 2] - start_x
        scale_x = size[0] / mid_width
    else:
        new_cx = intrinsics[0, 2]
        scale_x = size[0] / w
    scale_y = size[1] / h
    new_intrinsics = np.array([
        [intrinsics[0, 0] * scale_x, 0, new_cx * scale_x],
        [0, intrinsics[1, 1] * scale_y, intrinsics[1, 2] * scale_y],
        [0, 0, 1]
    ])
    return resized, new_intrinsics


def collate_fn(batch):
    target_size = (704, 256)
    processed_data = [crop_and_resize(item[0], target_size, item[3], False) for item in batch]
    imgs = [item[0] for item in processed_data]
    intrinsics = [item[1] for item in processed_data]

    gt_T_to_camera = [item[2] for item in batch]
    pcs = []
    masks = []
    max_num_points = 0
    for item in batch:
        max_num_points = max(max_num_points, item[1].shape[0])
    for item in batch:
        pc = item[1]
        masks.append(np.concatenate([np.ones(pc.shape[0]), np.zeros(max_num_points - pc.shape[0])], axis=0))
        if pc.shape[0] < max_num_points:
            pc = np.concatenate([pc, np.full((max_num_points - pc.shape[0], pc.shape[1]), 999999)], axis=0)
        pcs.append(pc)

    return imgs, pcs, masks, gt_T_to_camera, intrinsics

def main():
    args = parse_args()
    print(args)
    num_epochs = args.num_epochs
    dataset_root = args.dataset_root
    log_dir = args.log_dir
    if args.label is not None:
        log_dir = os.path.join(log_dir, args.label)
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{log_dir}/{current_time}"
    model_save_dir = os.path.join(log_dir, "model")
    ckpt_save_dir = os.path.join(log_dir, "checkpoint")
    if not os.path.exists(ckpt_save_dir) or args.save_ckpt_per_epoches > 0:
        os.makedirs(ckpt_save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    bev_calib_dir = os.path.join(parent_dir, 'kitti-bev-calib')
    shutil.copytree(bev_calib_dir, os.path.join(log_dir, 'kitti-bev-calib'))
    
    writer = SummaryWriter(log_dir)
    dataset = KittiDataset(dataset_root)

    generator = torch.Generator().manual_seed(114514)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
        shuffle=False
    )

    deformable_choise = args.deformable > 0
    bev_encoder_choise = args.bev_encoder > 0
    xyz_only_choise = args.xyz_only > 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVCalib(
        deformable=deformable_choise,
        bev_encoder=bev_encoder_choise
    ).to(device)

    if args.pretrain_ckpt is not None:
        state_dict = torch.load(args.pretrain_ckpt, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'], strict=True)
        print(f"Load pretrain model from {args.pretrain_ckpt}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler_choice = args.scheduler > 0
    if scheduler_choice:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    train_noise = {
        "angle_range_deg": args.angle_range_deg if args.angle_range_deg is not None else 20,
        "trans_range": args.trans_range if args.trans_range is not None else 1.5,
    }

    eval_noise = {
        "angle_range_deg": args.eval_angle_range_deg if args.eval_angle_range_deg is not None else train_noise["angle_range_deg"],
        "trans_range": args.eval_trans_range if args.eval_trans_range is not None else train_noise["trans_range"],
    }

    for epoch in range(num_epochs):
        model.train()
        train_loss = {}
        out_init_loss_choice = False
        if epoch < 5:
            out_init_loss_choice = True # Output initial loss for the first 5 epochs
        for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(train_loader):
            gt_T_to_camera = np.array(gt_T_to_camera).astype(np.float32)
            init_T_to_camera, _, _ = generate_single_perturbation_from_T(gt_T_to_camera, angle_range_deg=train_noise["angle_range_deg"], trans_range=train_noise["trans_range"])
            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            if xyz_only_choise:
                pcs = np.array(pcs)[:, :, :3]
            pcs = torch.from_numpy(np.array(pcs)).float().to(device)
            gt_T_to_camera = torch.from_numpy(gt_T_to_camera).float().to(device)
            init_T_to_camera = torch.from_numpy(init_T_to_camera).float().to(device)
            post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
            intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)

            optimizer.zero_grad()
            # img, pc, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, cam_intrinsic
            T_pred, init_loss, loss = model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=out_init_loss_choice)
            total_loss = loss["total_loss"]
            total_loss.backward()
            optimizer.step()
            for key in loss.keys():
                if key not in train_loss.keys():
                    train_loss[key] = loss[key].item()
                else:
                    train_loss[key] += loss[key].item()
            
            if init_loss is not None:
                for key in init_loss.keys():
                    train_key = f"init_{key}"
                    if train_key not in train_loss.keys():
                        train_loss[train_key] = init_loss[key].item()
                    else:
                        train_loss[train_key] += init_loss[key].item()

            if batch_index % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_index+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}")

        if scheduler_choice:   
            scheduler.step()    
        
        for key in train_loss.keys():
            train_loss[key] /= len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss {key}: {train_loss[key]:.4f}")
            writer.add_scalar(f"Loss/train/{key}", train_loss[key], epoch)
        
        if epoch == num_epochs - 1 or (args.save_ckpt_per_epoches > 0 and (epoch + 1) % args.save_ckpt_per_epoches == 0):
            ckpt_path = os.path.join(ckpt_save_dir, f"ckpt_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_noise': train_noise,
                'eval_noise': eval_noise,
                'args': vars(args) 
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        train_loss = None
        init_loss = None
        loss = None

        if epoch % args.eval_epoches == 0:
            eval_trans_range = eval_noise["trans_range"]
            eval_angle_range = eval_noise["angle_range_deg"]
            model.eval()
            val_loss = {}
            with torch.no_grad():
                for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(val_loader):
                    # img, pc, depth_img, gt_T_to_camera, init_T_to_camera
                    gt_T_to_camera = np.array(gt_T_to_camera).astype(np.float32)
                    init_T_to_camera, ang_err, trans_err = generate_single_perturbation_from_T(gt_T_to_camera, angle_range_deg=eval_angle_range, trans_range=eval_trans_range)
                    resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                    if xyz_only_choise:
                        pcs = np.array(pcs)[:, :, :3]
                    pcs = torch.from_numpy(np.array(pcs)).float().to(device)
                    gt_T_to_camera = torch.from_numpy(gt_T_to_camera).float().to(device)
                    init_T_to_camera = torch.from_numpy(init_T_to_camera).float().to(device)
                    post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
                    intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)
                    T_pred, init_loss, loss = model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False)

                    for key in loss.keys():
                        val_key = key
                        if val_key not in val_loss.keys():
                            val_loss[val_key] = loss[key].item()
                        else:
                            val_loss[val_key] += loss[key].item()
                    if init_loss is not None:
                        for key in init_loss.keys():
                            val_key = f"init_{key}"
                            if val_key not in val_loss.keys():
                                val_loss[val_key] = init_loss[key].item()
                            else:
                                val_loss[val_key] += init_loss[key].item()

            for key in val_loss.keys():
                val_loss[key] /= len(val_loader)
                print(f"Epoch [{epoch+1}/{num_epochs}], {eval_angle_range}_{eval_trans_range} Validation Loss {key}: {val_loss[key]:.4f}")
                writer.add_scalar(f"Loss/val/{key}", val_loss[key], epoch)

            val_loss = None
            loss = None

    writer.close()
    print(f"Logs are saved at {log_dir}")


if __name__ == "__main__":
    main()
