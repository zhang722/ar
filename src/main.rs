use std::{error::Error, io::Read};

use opencv::{
    prelude::*,
    videoio::{self, VideoCaptureTrait},
    highgui,
    core,
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
    // intrinsic and distortion
    let mut file = std::fs::File::open("config.json").expect("Unable to open file");
    // 读取文件内容
    let mut content = String::new();
    file.read_to_string(&mut content).expect("Unable to read the file");

    let model: Model = serde_json::from_str(&content)?; 
    let k = model.intrinsic.transpose();
    let dist = model.distortion;


    // video
    let video_path = "./imgs/test.mp4";
    let mut capture = videoio::VideoCapture::from_file(video_path, videoio::CAP_ANY)?;
    
    if !capture.is_opened()? {
        panic!("Unable to open video file!");
    }

    let width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
    let fps = capture.get(videoio::CAP_PROP_FPS)?;

    let fourcc = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?;
    let output_filename = "output.avi";
    let mut video_writer = videoio::VideoWriter::new(output_filename, fourcc, fps, core::Size2i::new(width as i32, height as i32), true)?;

    highgui::named_window("Video", highgui::WINDOW_NORMAL)?;
    let mut frame = Mat::default();
    loop {
        capture.read(&mut frame)?;
        
        if frame.size()?.width <= 0 {
            break;
        }

        // process
        if let Err(e) = gui::process(&mut frame, &k, &dist) {
            println!("Error: {}", e);
            continue;
        }

        video_writer.write(&frame)?;
        // show
        highgui::imshow("Video", &frame)?;
        if highgui::wait_key(30)? >= 0 {
            break;
        }
    }


    Ok(())
}
