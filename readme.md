# This is a demo of AR in Rust using:
1. OpenCV
2. nalgebra
3. Levenberg-Marquardt algorithm

# Result
![](imgs/test.gif)
![](imgs/output.gif)

# Try it
## Dependencies
1. Rust toolchain 
2. C++ OpenCV Library installed

## Steps 
1. Clone this project
2. Cargo run

## Custom config
If you want to run this demo with your own video, please change the parameters in the config.json file:

- "video_path" : path of your video 
- "pattern" : the width and height of the chessboard
- "intrinsic" : the internal camera parameters
- "distortion" : the camera distortion parameters [k1, k2, p1, p2]