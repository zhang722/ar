use std::error::Error;

use opencv::{
    prelude::*,
    videoio::{self, VideoCaptureTrait},
    highgui,
    imgproc,
    core::{self, Point2f},
};

use nalgebra as na;

use crate::{detect, pose, optimize_with_lm};

pub fn resize_width(src: &Mat, width: i32) -> Result<Mat, Box<dyn Error>> {
     // Get the original image dimensions
    let original_width = src.cols();
    let original_height = src.rows();

    // Calculate the new dimensions while maintaining the aspect ratio
    let aspect_ratio = original_height as f64 / original_width as f64;
    let new_width = width;
    let new_height = (new_width as f64 * aspect_ratio) as i32;

    // Create a new Mat object to store the resized image
    let mut resized_image = Mat::default();
    
    // Resize the image
    imgproc::resize(
        &src,
        &mut resized_image,
        opencv::core::Size::new(new_width, new_height),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    Ok(resized_image)
}

pub fn imshow(title: &str, image: &Mat) -> Result<(), Box<dyn Error>> {
    // Define the desired width or height for the displayed image
    let display_width = 800; // You can also use a desired height instead

    let resized_image = resize_width(image, display_width)?;

    // Create a window to display the image
    highgui::named_window("Image Display", highgui::WINDOW_NORMAL)?;

    // Show the resized image in the created window
    highgui::imshow("Image Display", &resized_image)?;

    // Wait for a key press and close the window
    highgui::wait_key(0)?;

    Ok(())
}


pub fn process(frame: &mut Mat, k: &na::Matrix3<f64>, dist: &na::Vector4<f64>) -> Result<(), Box<dyn Error>> {
    // Load the image in color mode
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Check if the image is empty
    if gray.empty() {
        println!("Failed to open the image.");
        return Err(From::from("gray image is empty"));
    }
    let points = detect::detect_corners(&gray)?; 
    let world_points = pose::generate_world_points(0.02, (7, 5))?;

    for p in points.iter() {
        imgproc::circle(frame, core::Point2i::new(p.x as i32, p.y as i32), 5, core::Scalar::new(255.0, 0.0, 0.0, 255.0), 1, 
            imgproc::LINE_8, 0)?;
    }


    let h = 0.5 * pose::compute_h(&points, &world_points)?;
    let tf = pose::compute_tf(&h, &k)?;
    let mut tfs = vec![tf];
    let img_point_set = vec![points];

    optimize_with_lm::optimize_with_lm(k, dist, &mut tfs, &img_point_set, &world_points)?;
    let tf = tfs[0];
    
    let reproj_p = world_points.iter().map(|p| {
        pose::project(k, dist, &(tf * na::Point3::<f64>::new(p.x, p.y, 0.0)))
    }).collect::<Vec<_>>();
    for p in reproj_p {
        imgproc::circle(frame, core::Point2i::new(p.x as i32, p.y as i32), 5, core::Scalar::new(0.0, 255.0, 0.0, 255.0), 1, 
            imgproc::LINE_8, 0)?;
    }

    paint(frame, k, dist, &tf)?;

    Ok(())
}


fn paint(image: &mut Mat, k: &na::Matrix3<f64>, distortion: &na::Vector4<f64>, tf: &na::Isometry3<f64>) -> Result<(), Box<dyn Error>> {
    let project = |k: na::Matrix3<f64>, tf: na::Isometry3<f64>, world_point: na::Point3<f64>|
    -> na::Point2<f64> {
        let transed = tf * world_point;
        let xn = transed.x / transed.z;
        let yn = transed.y / transed.z;
        let fx = k[(0, 0)];
        let fy = k[(1, 1)];
        let cx = k[(0, 2)];
        let cy = k[(1, 2)];
        let k1 = distortion[0];
        let k2 = distortion[1];
        let p1 = distortion[2];
        let p2 = distortion[3];
        let rn2 = xn * xn + yn * yn;

        let x = fx * (xn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p1 * xn * yn + p2 * (rn2 + 2.0 * xn * xn)) + cx;
        let y = fy * (yn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p2 * xn * yn + p1 * (rn2 + 2.0 * yn * yn)) + cy;

        na::Point2::<f64>::new(x, y)
    };

    let correct_tf = |tf: na::Isometry3<f64>| -> na::Isometry3<f64> {
        if tf.translation.z < 0.0 {
            let r = tf.to_homogeneous().fixed_view::<3, 3>(0, 0) 
                *
                na::Matrix3::<f64>::new(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0); 
            na::Isometry3::<f64>::from_parts(tf.translation.inverse(), 
                na::UnitQuaternion::<f64>::from_rotation_matrix(&na::Rotation3::<f64>::from_matrix_unchecked(r)))
        } else {
            tf
        }

    };

    let tf = correct_tf(*tf);

    let model = generate_model()?;
    for pairs in model {
        let p1 = project(*k, tf, pairs.0);
        let p2 = project(*k, tf, pairs.1);
        imgproc::line(image,
            core::Point2i::new(p1.x as i32, p1.y as i32),
            core::Point2i::new(p2.x as i32, p2.y as i32),
            core::Scalar::new(0.0, 0.0, 255.0, 255.0), 3, imgproc::LINE_8, 0)?;
    }

    Ok(())
}


fn generate_model() -> Result<Vec<(na::Point3<f64>, na::Point3<f64>)>, Box<dyn Error>> {
    let len = 0.1;
    let bottom = vec![
        na::Point3::<f64>::new(0.0, 0.0, 0.0),
        na::Point3::<f64>::new(len, 0.0, 0.0),
        na::Point3::<f64>::new(len, len, 0.0),
        na::Point3::<f64>::new(0.0, len, 0.0),
    ];

    let top = vec![
        na::Point3::<f64>::new(0.0, 0.0, -len),
        na::Point3::<f64>::new(len, 0.0, -len),
        na::Point3::<f64>::new(len, len, -len),
        na::Point3::<f64>::new(0.0, len, -len),
    ];
    
    let mut model = Vec::new();
    for idx in 0..4usize {
        if idx == 3 {
            model.push((bottom[idx], bottom[0]));
            model.push((top[idx], top[0]));
        } else {
            model.push((bottom[idx], bottom[idx + 1]));
            model.push((top[idx], top[idx + 1]));
        }
        model.push((bottom[idx], top[idx]));
    }

    Ok(model)
}


#[test]
fn test_opencv_video() -> Result<(), Box<dyn Error>> {
    let video_path = "./imgs/test.mp4";
    let mut capture = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    
    if !capture.is_opened()? {
        panic!("Unable to open video file!");
    }

    highgui::named_window("Video", highgui::WINDOW_NORMAL)?;
    let mut frame = Mat::default();

    loop {
        capture.read(&mut frame)?;
        
        if frame.size()?.width <= 0 {
            break;
        }

        highgui::imshow("Video", &frame)?;
        if highgui::wait_key(30)? >= 0 {
            break;
        }
    }

    Ok(())
}