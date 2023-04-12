use opencv::{
    prelude::*,
    imgproc,
    calib3d,
    core::{self, Point2f}, 
};

use nalgebra as na;

/// Detect chessboard corners using OpenCV binding library
/// Declaration:
/// '''
/// fn detect_corners(gray: &Mat) -> opencv::Result<Vec<core::Point2f>>
/// '''
/// Usage:
/// '''
/// let gray = ...;
/// let points = detect_corners(&gray).unwrap();
/// '''
/// gray: &opencv::core::Mat
/// return: opencv::Result<Vec<core::Point2f>>
pub fn detect_corners(gray: &Mat) -> opencv::Result<Vec<na::Point2<f64>>> {
    use opencv::core::Size;
    let patternsize: Size = Size::new(11, 8);
    let mut corners = Mat::default();
    let patternfound = calib3d::find_chessboard_corners(&gray, patternsize, &mut corners, 
        calib3d::CALIB_CB_ADAPTIVE_THRESH + calib3d::CALIB_CB_NORMALIZE_IMAGE
           + calib3d::CALIB_CB_FAST_CHECK)?;
    
    if patternfound {
        imgproc::corner_sub_pix(&gray, &mut corners, Size::new(11, 11), Size::new(-1, -1), 
            core::TermCriteria::new(core::TermCriteria_EPS + core::TermCriteria_MAX_ITER, 30, 0.1)?)?;
    }
    
    assert_eq!(corners.typ(), opencv::core::CV_32FC2);

    let mut points = Vec::new();

    for y in 0..corners.rows() {
        let point: Point2f = *corners.at(y)?;
        points.push(na::Point2::<f64>::new(point.x as f64, point.y as f64));
    }

    Ok(points)
}

