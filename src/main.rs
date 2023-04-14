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
pub struct Model {
    pub pattern: na::Vector2<i32>,
    pub intrinsic: na::Matrix3<f64>,
    pub distortion: na::Vector4<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    video_path: String,
    output_save: bool,
    output_save_path: String,
    model: Model,
}

fn main() -> Result<(), Box<dyn Error>> {
    // intrinsic and distortion
    let mut file = std::fs::File::open("config.json").expect("Unable to open file");
    // 读取文件内容
    let mut content = String::new();
    file.read_to_string(&mut content).expect("Unable to read the file");

    let config: Config = serde_json::from_str(&content)?; 
    let mut model = config.model;
    model.intrinsic = model.intrinsic.transpose();

    // video
    let video_path = config.video_path;
    let mut capture = videoio::VideoCapture::from_file(&video_path, videoio::CAP_ANY)?;
    
    if !capture.is_opened()? {
        panic!("Unable to open video file!");
    }

    let width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)?;
    let height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
    let fps = capture.get(videoio::CAP_PROP_FPS)?;

    let fourcc = videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?;
    let output_filename = &config.output_save_path;
    let mut video_writer = videoio::VideoWriter::new(output_filename, fourcc, fps, core::Size2i::new(width as i32, height as i32), true)?;

    highgui::named_window("Video", highgui::WINDOW_NORMAL)?;
    let mut frame = Mat::default();
    loop {
        capture.read(&mut frame)?;
        
        if frame.size()?.width <= 0 {
            break;
        }

        // process
        if let Err(e) = gui::process(&mut frame, &model) {
            println!("Error: {}", e);
            continue;
        }

        if config.output_save {
            video_writer.write(&frame)?;
        }

        // show
        highgui::imshow("Video", &frame)?;
        if highgui::wait_key(30)? >= 0 {
            break;
        }
    }


    Ok(())
}
