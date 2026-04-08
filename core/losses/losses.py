import torch
import torch.nn as nn
import torch.nn.functional as F
from .quat_tools import quaternion_distance, batch_quat2mat, batch_tvector2mat, quaternion_from_matrix

class quat_norm_loss(nn.Module):
    def __init__(self):
        super(quat_norm_loss, self).__init__()
    
    def forward(self, pred_rotation):
        """
        Args:
            pred_rotation: (B, 4)
        Returns:
            loss: (1), rad
        """
        norm_square = torch.sum(pred_rotation ** 2, dim = 1) # (B,)
        loss = (norm_square - 1) ** 2 # (B,)
        return loss.mean()

class translation_loss(nn.Module):
    def __init__(self, l1 = True):
        super(translation_loss, self).__init__()
        self.l1 = l1
        if l1:
            self.criterion = nn.SmoothL1Loss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='none')
    
    def forward(self, pred_translation, gt_translation):
        """
        Args:
            pred_translation: (B, 3)
            gt_translation: (B, 3)
        """
        if self.l1:
            loss = self.criterion(pred_translation, gt_translation)
            loss = loss.sum(dim = 1).mean()
        else:
            loss = self.criterion(pred_translation, gt_translation)
            loss = loss.sum(dim = 1)
            loss = torch.sqrt(loss).mean()
        return loss
    
class rotation_loss(nn.Module):
    def __init__(self):
        super(rotation_loss, self).__init__()
    
    def forward(self, pred_rotation, gt_rotation):
        """
        Args:
            pred_rotation: (B, 3, 3)
            gt_rotation: (B, 3, 3)
        Returns:
            loss: (1), rad
        """
        B, _, _ = pred_rotation.shape
        pred_rot = torch.zeros(B, 4, device = pred_rotation.device)
        gt_rot = torch.zeros(B, 4, device = pred_rotation.device)
        for i in range(B):
            pred_rot[i] = quaternion_from_matrix(pred_rotation[i])
            gt_rot[i] = quaternion_from_matrix(gt_rotation[i])
        loss = quaternion_distance(pred_rot, gt_rot, device = pred_rot.device)
        return loss.mean()

class PC_reproj_loss(nn.Module):
    def __init__(self):
        super(PC_reproj_loss, self).__init__()
    
    def forward(self, pcs, gt_T_to_camera, pred_translation, pred_rotation, mask = None):
        """
        Args:
            pcs: (B, N, 3)
            gt_T_to_camera: (B, 4, 4)
            init_T_to_camera: (B, 4, 4)
            pred_translation: (B, 3), which is T_pred^{-1} * T_init
            pred_rotation: (B, 3, 3), which is T_pred^{-1} * T_init
            mask: (B, N), where N is the maximum number of points in a batch
        Returns:
            reproj_loss: ||T_gt^{-1} * T_pred^{-1} * T_init * pcs - pcs||_2
        """
        B, N, _ = pcs.shape
        loss = torch.tensor(0.0, device = pcs.device)
        for i in range(B):
            RT_gt = gt_T_to_camera[i]
            T_pred = torch.eye(4, device = pcs.device)
            T_pred[:3, :3] = pred_rotation[i]
            T_pred[:3, 3] = pred_translation[i]
            RT_total = torch.matmul(RT_gt.inverse(), T_pred)
            pc = pcs[i]
            if mask is not None:
                pc = pc[mask[i] == 1]
            ones = torch.ones(pc.shape[0], 1, device=pc.device)
            points_h = torch.cat([pc, ones], dim = 1) # (N, 4)
            points_transformed = torch.matmul(points_h, RT_total.t())[:, :3] # (N, 3)
            error = (points_transformed - pc).norm(dim = 1)
            loss += error.mean()
        
        return loss / B
            
class realworld_loss(nn.Module):
    def __init__(self, weight_translation = 1.0, weight_quat_norm = 0.5, weight_rotation = 0.5, weight_PCreproj = 0.5, 
                 weight_bev_reproj = 0.5, weight_feat_align = 1.0, l1 = False):
        super(realworld_loss, self).__init__()
        self.weight_translation = weight_translation
        self.weight_rotation = weight_rotation
        self.weight_PCreproj = weight_PCreproj
        self.weight_quat_norm = weight_quat_norm
        self.weight_bev_reproj = weight_bev_reproj
        self.weight_feat_align = weight_feat_align
        self.translation_loss = translation_loss(l1 = True)
        self.real_translation_loss = translation_loss(l1 = False)
        self.rotation_loss = rotation_loss()
        self.quat_norm_loss = quat_norm_loss()
        self.PC_reproj_loss = PC_reproj_loss()
    
    def forward(self, pred_translation, pred_rotation, pcs, gt_T_to_camera, init_T_to_camera, mask = None):
        """
        Args:
            pcs: (B, N, 3)
            gt_T_to_camera: (B, 4, 4)
            init_T_to_camera: (B, 4, 4)
            pred_translation: (B, 3)
            pred_rotation: (B, 4)
            mask: (B, N), where N is the maximum number of points in a batch
        Expected:
            T_gt_expected = T_pred^{-1} * T_init
            T_pred = T_init * T_gt^{-1}
        Returns:
            total loss
        """
        T_pred = batch_tvector2mat(pred_translation)
        quat_norm_loss = self.quat_norm_loss(pred_rotation)
        R_pred = batch_quat2mat(pred_rotation)
        T_pred = torch.bmm(T_pred, R_pred) # (B, 4, 4)
        T_gt_expected = torch.matmul(T_pred.inverse(), init_T_to_camera)
        pred_translation = T_gt_expected[:, :3, 3]
        pred_rotation = T_gt_expected[:, :3, :3]
        gt_translation = gt_T_to_camera[:, :3, 3]
        gt_rotation = gt_T_to_camera[:, :3, :3]

        translation_loss = self.translation_loss(pred_translation, gt_translation)
        rotation_loss = self.rotation_loss(pred_rotation, gt_rotation)
        PC_reproj_loss = self.PC_reproj_loss(pcs, gt_T_to_camera, pred_translation, pred_rotation, mask)
        loss = self.weight_translation * translation_loss \
                + self.weight_rotation * rotation_loss \
                + self.weight_PCreproj * PC_reproj_loss \
                + self.weight_quat_norm * quat_norm_loss

        with torch.no_grad():
            real_trans_loss = self.real_translation_loss(pred_translation, gt_translation)

        ret = {
            "total_loss" : loss,
            "translation_loss" : real_trans_loss, # l1, m
            "rotation_loss" : rotation_loss / torch.pi * 180.0, # degree
            "quat_norm_loss" : quat_norm_loss, 
            "PC_reproj_loss" : PC_reproj_loss,
        }
        return ret, T_gt_expected