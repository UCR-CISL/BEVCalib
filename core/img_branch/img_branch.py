import torch
import torch.nn as nn
from .bev_pool.bev_pool import bev_pool
from .img_encoders import SwinT_tiny_Encoder


from core.proj_head import ProjectionHead
from core.bev_settings import xbound, ybound, zbound, down_ratio, sparse_shape, vsize_xyz, d_conf

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx

class LSS(nn.Module):
    ### Adapted from https://github.com/nv-tlabs/lift-splat-shoot
    def __init__(self, 
                 transformedImgShape = (3, 256, 704),
                 featureShape = (256, 32, 88),
                 d_conf = d_conf,
                 out_channels = 128,
                 ):
        super(LSS, self).__init__()
        _, self.orfH, self.orfW = transformedImgShape
        self.fC, self.fH, self.fW = featureShape
        self.d_st, self.d_end, self.d_step = d_conf
        self.D = torch.arange(self.d_st, self.d_end, self.d_step, dtype = torch.float).shape[0]
        self.out_channels = out_channels
        self.frustum = self.create_frustum()
        self.depth_net = nn.Sequential(
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.out_channels + self.D, 1),
        )
    
    def create_frustum(self):
        ds = torch.arange(self.d_st, self.d_end, self.d_step, dtype = torch.float).view(-1, 1, 1).expand(-1, self.fH, self.fW) # (D, fH, fW)
        xs = torch.linspace(0, self.orfW - 1, self.fW).view(1, 1, self.fW).expand(self.D, self.fH, self.fW)
        ys = torch.linspace(0, self.orfH - 1, self.fH).view(1, self.fH, 1).expand(self.D, self.fH, self.fW)

        frustum = torch.stack((xs, ys, ds), -1) # img space, the frustum is a cuboid.
        return nn.Parameter(frustum, requires_grad = False)

    def get_geometry(self, 
                     cam2ego_rot,
                     cam2ego_trans,
                     cam_intrins,
                     post_cam2ego_rot,
                     post_cam2ego_trans,
                     ):
        img_rots, img_trans, cam_intrins, img_post_rots, img_post_trans = cam2ego_rot, cam2ego_trans, cam_intrins, post_cam2ego_rot, post_cam2ego_trans
        B, N, _ = img_trans.shape
        points = self.frustum - img_post_trans.view(B, N, 1, 1, 1, 3) # (B, N, D, fH, fW, 3)
        points = torch.inverse(img_post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # before : (B, N, D, fH, fW, (x, y, z), 1), image space
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], # x * z, y * z
                points[:, :, :, :, :, 2:3] # z
            ),
            5,
        )
        # after : (B, N, D, fH, fW, (x * z, y * z, z), 1), transfrom from image space to camera space, the frustum transforms from cuboid to pyramid. 

        combine = img_rots.matmul(torch.inverse(cam_intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += img_trans.view(B, N, 1, 1, 1, 3) # To ego space

        return points 

    def get_cam_feature(self, img_feats):
        B, N, C, H, W = img_feats.shape
        img_feats = img_feats.view(B * N, C, H, W) 
        img_feats = self.depth_net(img_feats) # (B * N, out_channels + D, H, W)
        depth = img_feats[:, :self.D].softmax(dim = 1) # (B * N, D, H, W)
        img_feats = depth.unsqueeze(1) * img_feats[:, self.D : self.D + self.out_channels].unsqueeze(2) # (BN, 1, D, H, W) * (BN, out_channels, 1, H, W) => (BN, out_channels, D, H, W)
        img_feats = img_feats.view(B, N, self.out_channels, self.D, H, W)
        img_feats = img_feats.permute(0, 1, 3, 4, 5, 2)
        
        return img_feats

    def forward(self, 
                cam2ego_rot,
                cam2ego_trans,
                cam_intrins,
                post_cam2ego_rot,
                post_cam2ego_trans,
                img_feats
                ):
        geometry = self.get_geometry(cam2ego_rot=cam2ego_rot, cam2ego_trans=cam2ego_trans, cam_intrins=cam_intrins, post_cam2ego_rot=post_cam2ego_rot, post_cam2ego_trans=post_cam2ego_trans)
        img_depth_feature = self.get_cam_feature(img_feats)
        return geometry, img_depth_feature

class GaussianLSS(nn.Module):
    ### Adapted from https://github.com/nv-tlabs/lift-splat-shoot
    def __init__(self, 
                 transformedImgShape = (3, 256, 704),
                 featureShape = (256, 32, 88),
                 d_conf = d_conf,
                 out_channels = 128,
                 ):
        super(GaussianLSS, self).__init__()
        _, self.orfH, self.orfW = transformedImgShape
        self.fC, self.fH, self.fW = featureShape
        self.d_st, self.d_end, self.d_step = d_conf
        self.D = torch.arange(self.d_st, self.d_end, self.d_step, dtype = torch.float).shape[0]
        self.out_channels = out_channels
        self.frustum = self.create_frustum()
        self.depth_net = nn.Sequential(
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.fC, 3, padding=1),
            nn.BatchNorm2d(self.fC, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
            nn.Conv2d(self.fC, self.out_channels + self.D, 1),
        )
    
    def create_frustum(self):
        ds = torch.arange(self.d_st, self.d_end, self.d_step, dtype = torch.float).view(-1, 1, 1).expand(-1, self.fH, self.fW) # (D, fH, fW)
        xs = torch.linspace(0, self.orfW - 1, self.fW).view(1, 1, self.fW).expand(self.D, self.fH, self.fW)
        ys = torch.linspace(0, self.orfH - 1, self.fH).view(1, self.fH, 1).expand(self.D, self.fH, self.fW)

        frustum = torch.stack((xs, ys, ds), -1) # img space, the frustum is a cuboid.
        return nn.Parameter(frustum, requires_grad = False)

    def get_geometry(self, 
                     cam2ego_rot,
                     cam2ego_trans,
                     cam_intrins,
                     post_cam2ego_rot,
                     post_cam2ego_trans,
                     ):
        img_rots, img_trans, cam_intrins, img_post_rots, img_post_trans = cam2ego_rot, cam2ego_trans, cam_intrins, post_cam2ego_rot, post_cam2ego_trans
        B, N, _ = img_trans.shape
        points = self.frustum - img_post_trans.view(B, N, 1, 1, 1, 3) # (B, N, D, fH, fW, 3)
        points = torch.inverse(img_post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # before : (B, N, D, fH, fW, (x, y, z), 1), image space
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], # x * z, y * z
                points[:, :, :, :, :, 2:3] # z
            ),
            5,
        )
        # after : (B, N, D, fH, fW, (x * z, y * z, z), 1), transfrom from image space to camera space, the frustum transforms from cuboid to pyramid. 

        combine = img_rots.matmul(torch.inverse(cam_intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += img_trans.view(B, N, 1, 1, 1, 3) # To ego space

        return points 

    def get_cam_feature(self, img_feats):
        B, N, C, H, W = img_feats.shape
        img_feats = img_feats.view(B * N, C, H, W) 
        img_feats = self.depth_net(img_feats) # (B * N, out_channels + D, H, W)
        depth = img_feats[:, :self.D].softmax(dim = 1) # (B * N, D, H, W)
        img_feats = depth.unsqueeze(1) * img_feats[:, self.D : self.D + self.out_channels].unsqueeze(2) # (BN, 1, D, H, W) * (BN, out_channels, 1, H, W) => (BN, out_channels, D, H, W)
        img_feats = img_feats.view(B, N, self.out_channels, self.D, H, W)
        img_feats = img_feats.permute(0, 1, 3, 4, 5, 2)
        
        return img_feats

    def forward(self, 
                cam2ego_rot,
                cam2ego_trans,
                cam_intrins,
                post_cam2ego_rot,
                post_cam2ego_trans,
                img_feats
                ):
        geometry = self.get_geometry(cam2ego_rot=cam2ego_rot, cam2ego_trans=cam2ego_trans, cam_intrins=cam_intrins, post_cam2ego_rot=post_cam2ego_rot, post_cam2ego_trans=post_cam2ego_trans)
        img_depth_feature = self.get_cam_feature(img_feats)
        return geometry, img_depth_feature

class Cam2BEV(nn.Module):
    def __init__(self, 
                 output_indices = [1, 2, 3], 
                 featureShape = (256, 32, 88), 
                 encoder_out_channels = 256,
                 FPN_in_channels = [192, 384, 768], 
                 FPN_out_channels = 256,
                 ):
        super(Cam2BEV, self).__init__()
        self.lss = LSS()
        self.CamEncode = SwinT_tiny_Encoder(output_indices, featureShape, encoder_out_channels, FPN_in_channels, FPN_out_channels)
        dx, bx, nx = gen_dx_bx(xbound=xbound, ybound=ybound, zbound=zbound)
        self.dx = nn.Parameter(dx, requires_grad = False)
        self.bx = nn.Parameter(bx, requires_grad = False)
        self.nx = nn.Parameter(nx, requires_grad = False)
        self.proj_head = ProjectionHead(embedding_dim=self.lss.out_channels)
        self.out_channels = self.proj_head.projection_dim
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))
        print(f"cam bev resolution: {nx[0].item()} x {nx[1].item()} x {nx[2].item()}")

    def bev_pool(self, img_depth_feature, geometry):
        img_pc = img_depth_feature
        geom_feats = geometry
        B, N, D, H, W, C = img_pc.shape
        Nprime = B * N * D * H * W
        img_pc = img_pc.reshape(Nprime, C)

        # Align the geometry to the voxel grid
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=img_pc.device, dtype=torch.long)
                for ix in range(B)
            ]
        )   
        geom_feats = torch.cat([geom_feats, batch_ix], 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        img_pc = img_pc[kept]
        geom_feats = geom_feats[kept]

        try:
            cam_bev = bev_pool(img_pc, geom_feats, B, self.nx[2], self.nx[0], self.nx[1]) # (B, out_channels, nx[2], nx[0], nx[1])
        except:
            cam_bev = torch.zeros(B, C, self.nx[2].item(), self.nx[0].item(), self.nx[1].item(), device = img_pc.device, dtype = img_pc.dtype)

        cam_bev = torch.cat(cam_bev.unbind(dim = 2), 1)

        return cam_bev
    
    def ref_bev_pool(self, img_depth_feature, geometry):
        with torch.no_grad():
            img_pc = img_depth_feature
            geom_feats = geometry
            B, N, D, H, W, C = img_pc.shape
            Nprime = B * N * D * H * W
            img_pc = img_pc.reshape(Nprime, C)

            # Align the geometry to the voxel grid
            geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
            geom_feats = geom_feats.view(Nprime, 3)
            batch_ix = torch.cat(
                [
                    torch.full([Nprime // B, 1], ix, device=img_pc.device, dtype=torch.long)
                    for ix in range(B)
                ]
            )   
            geom_feats = torch.cat([geom_feats, batch_ix], 1)

            # filter out points that are outside box
            kept = (
                (geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2])
            )
            img_pc = img_pc[kept]
            geom_feats = geom_feats[kept]
            try:
                cam_bev = bev_pool(img_pc, geom_feats, B, self.nx[2], self.nx[0], self.nx[1]) # (B, out_channels, nx[2], nx[0], nx[1])
            except:
                cam_bev = torch.zeros(B, C, self.nx[2].item(), self.nx[0].item(), self.nx[1].item(), device = img_pc.device, dtype = img_pc.dtype)

            binary_bev = torch.max(cam_bev, dim=2)[0]

            binary_bev = (binary_bev != 0).float()

        return binary_bev

    def forward(self, 
                cam2ego_T,
                cam_intrins,
                post_cam2ego_T,
                imgs
                ):
        """
        Args:
            cam2ego_T: (B, N, 4, 4), transformation matrix from camera to ego.
            cam_intrins: (B, 3, 3), camera intrinsic matrix.
            post_cam2ego_T: (B, N, 4, 4), transformation matrix from camera to ego after data aug.
            imgs: (B, N, 3, H, W), original image (with data aug).
        Returns:
            bev_feats: (B, C, H, W), bird eye view features.
            cam_bev_mask: (B, H, W), mask for bev_feats. Noting cam_bev_mask[:, i, :, :] is the same for all i.
        """
        cam2ego_rot = cam2ego_T[:, :, :3, :3]
        cam2ego_trans = cam2ego_T[:, :, :3, 3]
        post_cam2ego_rot = post_cam2ego_T[:, :, :3, :3]
        post_cam2ego_trans = post_cam2ego_T[:, :, :3, 3]

        imgs = (imgs - self.mean) / self.std
        img_feats = self.CamEncode(imgs) 
        geometry, img_depth_feature = self.lss(cam2ego_rot=cam2ego_rot, cam2ego_trans=cam2ego_trans, cam_intrins=cam_intrins, post_cam2ego_rot=post_cam2ego_rot, post_cam2ego_trans=post_cam2ego_trans, img_feats=img_feats)
        bev_feats = self.bev_pool(geometry=geometry, img_depth_feature=img_depth_feature)
        with torch.no_grad():
            geom_detach = geometry.detach()
            ones_feat = torch.ones_like(img_depth_feature).to(img_depth_feature.device).detach()
            ref_feats = self.ref_bev_pool(geometry=geom_detach, img_depth_feature=ones_feat)
        B, C, H, W = bev_feats.shape
        cam_bev_mask = ref_feats != 0
        ref_mask = cam_bev_mask[:, 0:1, :, :]
        try:
            assert torch.all(cam_bev_mask == ref_mask) # The masks should be same for different channels.
        except:
            print(f"ERROR: cam_bev_mask != ref_mask")
            torch.save(cam2ego_T, "cam2ego_T.pt")
            torch.save(cam_intrins, "cam_intrins.pt")
            torch.save(post_cam2ego_T, "post_cam2ego_T.pt")
            torch.save(imgs, "imgs.pt")

        bev_feats = bev_feats.permute(0, 2, 3, 1).reshape(B*H*W, C)
        bev_feats = self.proj_head(bev_feats)
        bev_feats = bev_feats.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return bev_feats, cam_bev_mask[:, 0, :, :]
    
def generate_random_rt_matrix(batch_size=1, r_range=(-torch.pi, torch.pi), t_range=(-1, 1)):
    rx = torch.rand(batch_size, 1) * (r_range[1] - r_range[0]) + r_range[0]
    ry = torch.rand(batch_size, 1) * (r_range[1] - r_range[0]) + r_range[0]
    rz = torch.rand(batch_size, 1) * (r_range[1] - r_range[0]) + r_range[0]
    
    zeros = torch.zeros(batch_size, 1)
    ones = torch.ones(batch_size, 1)
    
    Rx = torch.stack([
        torch.cat([ones, zeros, zeros], dim=1),
        torch.cat([zeros, torch.cos(rx), -torch.sin(rx)], dim=1),
        torch.cat([zeros, torch.sin(rx), torch.cos(rx)], dim=1)
    ], dim=2)
    
    Ry = torch.stack([
        torch.cat([torch.cos(ry), zeros, torch.sin(ry)], dim=1),
        torch.cat([zeros, ones, zeros], dim=1),
        torch.cat([-torch.sin(ry), zeros, torch.cos(ry)], dim=1)
    ], dim=2)
    
    Rz = torch.stack([
        torch.cat([torch.cos(rz), -torch.sin(rz), zeros], dim=1),
        torch.cat([torch.sin(rz), torch.cos(rz), zeros], dim=1),
        torch.cat([zeros, zeros, ones], dim=1)
    ], dim=2)
    
    R = torch.bmm(torch.bmm(Rz, Ry), Rx)
    t = torch.rand(batch_size, 3, 1) * (t_range[1] - t_range[0]) + t_range[0]
    
    RT = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    RT[:, :3, :3] = R
    RT[:, :3, 3:4] = t
    
    return RT.unsqueeze(1)

def test():
    B = 1
    N = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cam2ego_T = generate_random_rt_matrix(r_range=(-torch.pi/4, torch.pi/4), t_range=(-1, 1)).to(device)
    cam_intrinsic = torch.tensor([[1.7489e+03, 0.0000e+00, 9.8110e+02],
        [0.0000e+00, 1.7556e+03, 5.4997e+02],
        [0.0000e+00, 0.0000e+00, 1.0000e+00]]).to(device)
    post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(B, N, 1, 1).float().to(device)
    imgs = torch.randn(B, N, 3, 400, 640).to(device)
    imgs = imgs / imgs.max() * 255
    model = Cam2BEV().to(device)
    bev_feats, cam_bev_mask = model(cam2ego_T=cam2ego_T, cam_intrins=cam_intrinsic, post_cam2ego_T=post_cam2ego_T, imgs=imgs)

if __name__ == "__main__":
    for i in range(1000):
        try:
            if i % 10 == 0:
                print(f"iter {i}")
            test()
        except:
            break