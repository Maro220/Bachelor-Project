from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
import numpy as np

# 1. Initialize the dataset (pointing to your existing v1.0-mini folder)
nusc = NuScenes(version='v1.0-mini', dataroot='data/v1.0-mini', verbose=True)

# 2. Grab the first scene and its first keyframe (sample)
my_scene = nusc.scene[0]
sample_token = my_scene['first_sample_token']
sample_record = nusc.get('sample', sample_token)

# 3. Get the token specifically for the Front Camera
cam_front_token = sample_record['data']['CAM_FRONT']

# 4. THE MAGIC STEP: The SDK automatically fetches the image path, the 3D boxes 
#    (already translated to the camera's coordinate frame), and the camera matrix.
data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam_front_token)

print(f"\nProcessing Image: {data_path}")
print("-" * 40)

# 5. Project the 3D corners to 2D pixels for your JSON schema
for box in boxes:
    # Get the 8 corners of the 3D box
    corners_3d = box.corners()
    
    # Project those 3D corners onto the 2D image plane using the camera matrix
    corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :]
    
    # Extract the standard 2D bounding box limits [xmin, ymin, xmax, ymax]
    xmin, xmax = corners_2d[0, :].min(), corners_2d[0, :].max()
    ymin, ymax = corners_2d[1, :].min(), corners_2d[1, :].max()
    
    # Filter out boxes that are completely behind the camera
    if np.any(corners_3d[2, :] < 0.1):
        continue
        
    print(f"Class: {box.name}")
    print(f"2D Bounding Box: [{xmin:.1f}, {ymin:.1f}, {xmax:.1f}, {ymax:.1f}]\n")