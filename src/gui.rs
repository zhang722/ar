use std::error::Error;

use opencv::{
    prelude::*,
    imgproc,
    core,
};

use nalgebra as na;

use crate::{detect, pose, optimize_with_lm};

pub fn process(frame: &mut Mat, model: &super::Model) -> Result<(), Box<dyn Error>> {
    let super::Model {intrinsic: k, distortion: dist, pattern} = model;
    let pattern = (pattern.x, pattern.y);

    // Load the image in color mode
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Check if the image is empty
    if gray.empty() {
        println!("Failed to open the image.");
        return Err(From::from("gray image is empty"));
    }
    let points = detect::detect_corners(&gray, pattern)?; 
    let world_points = pose::generate_world_points(0.02, pattern)?;

    for p in points.iter() {
        imgproc::circle(frame, core::Point2i::new(p.x as i32, p.y as i32), 5, core::Scalar::new(255.0, 0.0, 0.0, 255.0), 1, 
            imgproc::LINE_8, 0)?;
    }


    let h = 0.5 * pose::compute_h(&points, &world_points)?;
    let tf = pose::compute_tf(&h, k)?;
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
        pose::project(&k, distortion, &transed)
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


// Line = (start, end)
type Line = (na::Point3<f64>, na::Point3<f64>);

// Generate a model of a cube
fn generate_model() -> Result<Vec<Line>, Box<dyn Error>> {
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
