use std::error::Error;

use opencv::{
    prelude::*,
    highgui,
    imgproc,
};

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