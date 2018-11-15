# 3D_channel_SIFT
Registration routine based on SIFT algorithm.

This algorithm applies SIFT two consecutive times.
1. XY-SIFT: 
  - computes the maximum projections along the Z direction
  - find features and match them
  - applies the XY homography to the full stack
2. YZ-SIFT: using the
  - computes maximum projection along the X direction
  - expand the Z dimension N=4 times (to take into account anisotropic resolution)
  - find features and match them
  - applies theYZ homography the the YZ slice of the full stack (upon resampling to match resolution of the max proj)
  - downsample registered slice to match original resolution
