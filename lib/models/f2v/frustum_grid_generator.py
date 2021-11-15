import torch
import torch.nn as nn
import kornia


def project_to_image(project, points):
    """
    Project points to image
    Args:
        project [torch.tensor(..., 3, 4)]: Projection matrix
        points [torch.Tensor(..., 3)]: 3D points
    Returns:
        points_img [torch.Tensor(..., 2)]: Points in image
        points_depth [torch.Tensor(...)]: Depth of each point
    """
    # Reshape tensors to expected shape
    points = kornia.convert_points_to_homogeneous(points)
    points = points.unsqueeze(dim=-1)
    project = project.unsqueeze(dim=1)

    # Transform points to image and get depths
    points_t = project @ points
    points_t = points_t.squeeze(dim=-1)
    points_img = kornia.convert_points_from_homogeneous(points_t)
    points_depth = points_t[..., -1] - project[..., 2, 3]

    return points_img, points_depth

def normalize_coords(coords, shape):
    """
    Normalize coordinates of a grid between [-1, 1]
    Args:
        coords [torch.Tensor(..., 2)]: Coordinates in grid
        shape [torch.Tensor(2)]: Grid shape [H, W]
    Returns:
        norm_coords [torch.Tensor(.., 2)]: Normalized coordinates in grid
    """
    min_n = -1
    max_n = 1
    shape = torch.flip(shape, dims=[0])  # Reverse ordering of shape #[1242,  375,   80]

    # Subtract 1 since pixel indexing from [0, shape - 1]
    norm_coords = coords / (shape - 1) * (max_n - min_n) + min_n
    return norm_coords

def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices


class FrustumGridGenerator(nn.Module):

    def __init__(self, grid_size, pc_range, disc_cfg):
        """
        Initializes Grid Generator for frustum features
        Args:
            grid_size [np.array(3)]: Voxel grid shape [X, Y, Z]
            pc_range [list]: Voxelization point cloud range [X_min, Y_min, Z_min, X_max, Y_max, Z_max]
            disc_cfg [int]: Depth discretiziation configuration
        """
        super().__init__()
        self.dtype = torch.float32
        self.grid_size = torch.as_tensor(grid_size) #[280, 376, 25]
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg

        # Calculate voxel size
        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size #[0.1600, 0.1600, 0.1600]

        # Create voxel grid
        self.depth, self.width, self.height = self.grid_size.int()
        self.voxel_grid = kornia.utils.create_meshgrid3d(depth=self.depth,
                                                         height=self.height,
                                                         width=self.width,
                                                         normalized_coordinates=False) #[1, 280, 25, 376, 3]

        self.voxel_grid = self.voxel_grid.permute(0, 1, 3, 2, 4)  #[1, 280, 25, 376, 3]-->#[1, 280, 376, 25, 3]

        # Add offsets to center of voxel
        self.voxel_grid += 0.5
        self.grid_to_lidar = self.grid_to_lidar_unproject(pc_min=self.pc_min,
                                                          voxel_size=self.voxel_size)

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        """
        Calculate grid to LiDAR unprojection for each plane
        Args:
            pc_min [torch.Tensor(3)]: Minimum of point cloud range [X, Y, Z] (m)
            voxel_size [torch.Tensor(3)]: Size of each voxel [X, Y, Z] (m)
        Returns:
            unproject [torch.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
        """
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor([[x_size, 0, 0, x_min],
                                  [0, y_size, 0, y_min],
                                  [0,  0, z_size, z_min],
                                  [0,  0, 0, 1]],
                                 dtype=self.dtype)  # (4, 4)

        return unproject

    def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_cam, cam_to_img):
        """
        Transforms voxel sampling grid into frustum sampling grid
        Args:
            grid [torch.Tensor(B, X, Y, Z, 3)]: Voxel sampling grid
            grid_to_lidar [torch.Tensor(4, 4)]: Voxel grid to LiDAR unprojection matrix
            lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
        Returns:
            frustum_grid [torch.Tensor(B, X, Y, Z, 3)]: Frustum sampling grid
        """
        B = lidar_to_cam.shape[0]

        # Create transformation matricies
        V_G = grid_to_lidar  # Voxel Grid -> LiDAR (4, 4)
        C_V = lidar_to_cam  # LiDAR -> Camera (B, 4, 4)
        I_C = cam_to_img  # Camera -> Image (B, 3, 4)
        trans = C_V @ V_G

        # Reshape to match dimensions
        trans = trans.reshape(B, 1, 1, 4, 4)
        voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)

        # Transform to camera frame
        camera_grid = kornia.transform_points(trans_01=trans, points_1=voxel_grid) #[B, 280, 376, 25, 3]

        # Project to image
        I_C = I_C.reshape(B, 1, 1, 3, 4)
        image_grid, image_depths = project_to_image(project=I_C, points=camera_grid)
        #image_grid: [B, 280, 376, 25, 2]
        #image_depths: [B, 280, 376, 25]

        # Convert depths to depth bins
        image_depths = bin_depths(depth_map=image_depths, **self.disc_cfg) #[B, 280, 376, 25] return index

        # Stack to form frustum grid
        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        return frustum_grid

    def forward(self, lidar_to_cam, cam_to_img, image_shape):
        """
        Generates sampling grid for frustum features
        Args:
            lidar_to_cam [torch.Tensor(B, 4, 4)]: LiDAR to camera frame transformation
            cam_to_img [torch.Tensor(B, 3, 4)]: Camera projection matrix
            image_shape [torch.Tensor(B, 2)]: Image shape [H, W]
        Returns:
            frustum_grid [torch.Tensor(B, X, Y, Z, 3)]: Sampling grids for frustum features
        """

        frustum_grid = self.transform_grid(voxel_grid=self.voxel_grid.to(lidar_to_cam.device),
                                           grid_to_lidar=self.grid_to_lidar.to(lidar_to_cam.device),
                                           lidar_to_cam=lidar_to_cam,
                                           cam_to_img=cam_to_img)
        #frustum_grid: [B, 280, 376, 25, 3]
        # Normalize grid
        image_shape, _ = torch.max(image_shape, dim=0) #[ 375, 1242]

        image_depth = torch.tensor([self.disc_cfg["num_bins"]], device=image_shape.device, dtype=image_shape.dtype)
        frustum_shape = torch.cat((image_depth, image_shape)) #[80, 375, 1242]

        frustum_grid = normalize_coords(coords=frustum_grid, shape=frustum_shape) #[B, 280, 376, 25, 3]

        # Replace any NaNs or infinites with out of bounds
        mask = ~torch.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid
