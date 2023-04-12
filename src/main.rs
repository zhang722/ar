use std::{error::Error, io::Read};

use opencv::{
    prelude::*,
    imgcodecs,
    imgproc,
    core::{self}, 
};

use nalgebra as na;

mod detect;
mod gui;
mod pose;
mod lm;
mod optimize_with_lm;

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct Model {
    intrinsic: na::Matrix3<f64>,
    distortion: na::Vector4<f64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let image_path = "imgs/100000.png"; // Replace with your image path
    // Load the image in color mode
    let mut image = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    let mut gray = Mat::default();
    imgproc::cvt_color(&image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

    // Check if the image is empty
    if gray.empty() {
        println!("Failed to open the image.");
        return Err(From::from("gray image is empty"));
    }

    let points = detect::detect_corners(&gray)?; 
    let world_points = pose::generate_world_points(0.02, (11, 8))?;

    for (idx, p) in points.iter().enumerate() {
        imgproc::circle(&mut image, core::Point2i::new(p.x as i32, p.y as i32), 5, core::Scalar::new(255.0, 0.0, 0.0, 255.0), 1, 
            imgproc::LINE_8, 0)?;
    }

    // 打开文件
    let mut file = std::fs::File::open("config.json").expect("Unable to open file");

    // 读取文件内容
    let mut content = String::new();
    file.read_to_string(&mut content).expect("Unable to read the file");

    let model: Model = serde_json::from_str(&content)?; 
    let k = model.intrinsic.transpose();
    let dist = model.distortion;


    let h = 0.6 * pose::compute_h(&points, &world_points)?;
    let mut tf = pose::compute_tf(&h, &k)?;
    let mut tfs = vec![tf];
    let img_point_set = vec![points];

    optimize_with_lm::optimize_with_lm(&k, &dist, &mut tfs, &img_point_set, &world_points)?;
    let tf = tfs[0];
    
    let reproj_p = world_points.iter().map(|p| {
        pose::project(&k, &dist, &(tf * na::Point3::<f64>::new(p.x, p.y, 0.0)))
    }).collect::<Vec<_>>();
    for p in reproj_p {
        println!("{}", p);
        imgproc::circle(&mut image, core::Point2i::new(p.x as i32, p.y as i32), 5, core::Scalar::new(0.0, 255.0, 0.0, 255.0), 1, 
            imgproc::LINE_8, 0)?;
    }


    gui::imshow("title", &image)?;

    Ok(())
}