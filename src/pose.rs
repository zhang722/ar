use std::error::Error;

use opencv::{
    core::{Size}, imgcodecs, prelude::{Mat, MatTraitConst}, imgproc, 
};
use nalgebra as na;


type NormalizedPoints= Vec<na::Point2<f64>>;
type NormMatrix = na::Matrix3<f64>;
/// Normalize the points and return the normalized points and the normalization matrix
fn normalize(point_vec: &Vec<na::Point2<f64>>) 
-> Result<(NormalizedPoints, NormMatrix), Box<dyn Error>>
{
    let mut norm_t = na::Matrix3::<f64>::identity();
    let mut normed_point_vec = Vec::new();
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for p in point_vec {
        mean_x += p.x;
        mean_y += p.y;
    }
    mean_x /= point_vec.len() as f64;
    mean_y /= point_vec.len() as f64;
    let mut mean_dev_x = 0.0;
    let mut mean_dev_y = 0.0;
    for p in point_vec {
        mean_dev_x += (p.x - mean_x).abs();
        mean_dev_y += (p.y - mean_y).abs();
    }
    mean_dev_x /= point_vec.len() as f64;
    mean_dev_y /= point_vec.len() as f64;
    let sx = 1.0 / mean_dev_x;
    let sy = 1.0 / mean_dev_y;

    for p in point_vec {
        let mut p_tmp = na::Point2::<f64>::new(0.0, 0.0);
        p_tmp.x = sx * p.x - mean_x * sx;
        p_tmp.y = sy * p.y - mean_y * sy;
        normed_point_vec.push(p_tmp);
    }
    norm_t[(0, 0)] = sx;
    norm_t[(0, 2)] = -mean_x * sx;
    norm_t[(1, 1)] = sy;
    norm_t[(1, 2)] = -mean_y * sy;

    Ok((normed_point_vec, norm_t))
}

/// Generate the world points in the chessboard
pub fn generate_world_points(square_length: f64, pattern: (i32, i32)) -> Result<Vec<na::Point2<f64>>, Box<dyn Error>> {
    let (width, height) = pattern;
    let mut world_points = Vec::new();
    for x in 0..height {
        for y in 0..width {
            world_points.push(na::Point2::<f64>::new(y as f64 * square_length, x as f64 * square_length));
        }
    }
    Ok(world_points)
}

/// Compute the homography matrix H
pub fn compute_h(img_points: &Vec<na::Point2<f64>>, world_points: &Vec<na::Point2<f64>>) -> Result<na::Matrix3::<f64>, Box<dyn Error>> {
    let num_points = img_points.len();
    assert_eq!(num_points, world_points.len());

    // at least 4 point if want to compute H
    assert!(num_points > 3);

    type MatrixXx9<T> = na::Matrix<T, na::Dyn, na::U9, na::VecStorage<T, na::Dyn, na::U9>>;
    type RowVector9<T> = na::Matrix<T, na::U1, na::U9, na::ArrayStorage<T, 1, 9>>;

    let norm_img = normalize(img_points)?;
    let norm_world = normalize(world_points)?;

    let mut a = MatrixXx9::<f64>::zeros(num_points * 2);

    let img_world_points_iter = norm_img.0.iter().zip(norm_world.0.iter());
    for (idx, (img_point, world_point)) in img_world_points_iter.enumerate() {
        let u = img_point.x;
        let v = img_point.y;
        let x_w = world_point.x;
        let y_w = world_point.y;

        let ax = RowVector9::<f64>::from_vec(vec![
            x_w, y_w, 1.0, 0.0, 0.0, 0.0, -u*x_w, -u*y_w, -u 
        ]);
        let ay = RowVector9::<f64>::from_vec(vec![
            0.0, 0.0, 0.0, x_w, y_w, 1.0, -v*x_w, -v*y_w, -v
        ]);
        
        a.set_row(2 * idx, &ax);
        a.set_row(2 * idx + 1, &ay);
    } 
    let svd = a.svd(true, true);
    let v_t = match svd.v_t {
        Some(v_t) => v_t,
        None => return Err(From::from("compute V failed")),
    };
    let last_row = v_t.row(8);

    // construct matrix from vector
    let mut ret = na::Matrix3::<f64>::from_iterator(last_row.into_iter().cloned()).transpose();

    ret = match norm_img.1.try_inverse() {
        Some(m) => m,
        None => return Err(From::from("compute inverse norm_img failed")),
    } * ret * norm_world.1;
    
    Ok(ret)  
}

/// Compute the transformation matrix T
pub fn compute_tf(h: &na::Matrix3<f64>, k: &na::Matrix3<f64>) -> Result<na::Isometry3<f64>, Box<dyn Error>>{
    let a = match k.try_inverse() {
        Some(m) => m, 
        None => return Err(From::from("k is not invertible")),
    } * h; 
    let r1 = a.column(0);
    let r2 = a.column(1);
    let r3 = r1.cross(&r2);
    let mut r = na::Matrix3::<f64>::zeros();
    r.set_column(0, &r1);
    r.set_column(1, &r2);
    r.set_column(2, &r3);
    
    let r = r.normalize();

    // get nearest rotation matrix from r since r may not be a rotation matrix
    let svd = r.svd(true, true);
    let r = 
    match svd.u {
        Some(u) => u,
        None => return Err(From::from("compute V failed")),
    }
    * 
    match svd.v_t {
        Some(v_t) => v_t,
        None => return Err(From::from("compute V failed")),
    }; 

    let r = na::Rotation3::<f64>::from_matrix_unchecked(r);
    let t = a.column(2);
    let tf = 
        na::Isometry3::<f64>::from_parts(na::Translation3::new(t[0], t[1], t[2]), na::UnitQuaternion::from_rotation_matrix(&r));

    Ok(tf)
}

/// Project a scene point $p_s$(scene) to image point $p_p$(pixel)
pub fn project(
    intrinsic: &na::Matrix3<f64>,  //fx, fy, cx, cy
    distortion: &na::Vector4<f64>, //k1, k2, p1, p2
    pt: &na::Point3<f64>,
) -> na::Point2<f64> {
    let fx = intrinsic[(0, 0)];
    let fy = intrinsic[(1, 1)];
    let cx = intrinsic[(0, 2)];
    let cy = intrinsic[(1, 2)];
    let k1 = distortion[0];
    let k2 = distortion[1];
    let p1 = distortion[2];
    let p2 = distortion[3];

    let xn = pt.x / pt.z;
    let yn = pt.y / pt.z;
    let rn2 = xn * xn + yn * yn;
    na::Point2::<f64>::new(
        fx * (xn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p1 * xn * yn + p2 * (rn2 + 2.0 * xn * xn)) + cx,
        fy * (yn * (1.0 + k1 * rn2 + k2 * rn2 * rn2) + 2.0 * p2 * xn * yn + p1 * (rn2 + 2.0 * yn * yn)) + cy
    )
}
