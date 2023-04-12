use nalgebra as na;
use crate::lm;


/// Produces a skew-symmetric or "cross-product matrix" from
/// a 3-vector. This is needed for the `exp_map` and `log_map`
/// functions
fn skew_sym(v: na::Vector3<f64>) -> na::Matrix3<f64> {
    let mut ss = na::Matrix3::zeros();
    ss[(0, 1)] = -v[2];
    ss[(0, 2)] = v[1];
    ss[(1, 0)] = v[2];
    ss[(1, 2)] = -v[0];
    ss[(2, 0)] = -v[1];
    ss[(2, 1)] = v[0];
    ss
}

/// Converts a 6-Vector Lie Algebra representation of a rigid body
/// transform to an NAlgebra Isometry (quaternion+translation pair)
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn exp_map(param_vector: &na::Vector6<f64>) -> na::Isometry3<f64> {
    let t = param_vector.fixed_view::<3, 1>(0, 0);
    let omega = param_vector.fixed_view::<3, 1>(3, 0);
    let theta = omega.norm();
    let half_theta = 0.5 * theta;
    let quat_axis = omega * half_theta.sin() / theta;
    let quat = if theta > 1e-6 {
        na::UnitQuaternion::from_quaternion(na::Quaternion::new(
            half_theta.cos(),
            quat_axis.x,
            quat_axis.y,
            quat_axis.z,
        ))
    } else {
        na::UnitQuaternion::identity()
    };

    let mut v = na::Matrix3::<f64>::identity();
    if theta > 1e-6 {
        let ssym_omega = skew_sym(omega.clone_owned());
        v += ssym_omega * (1.0 - theta.cos()) / (theta.powi(2))
            + (ssym_omega * ssym_omega) * ((theta - theta.sin()) / (theta.powi(3)));
    }

    let trans = na::Translation::from(v * t);

    na::Isometry3::from_parts(trans, quat)
}

/// Converts an NAlgebra Isometry to a 6-Vector Lie Algebra representation
/// of a rigid body transform.
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn log_map(input: &na::Isometry3<f64>) -> na::Vector6<f64> {
    let t: na::Vector3<f64> = input.translation.vector;

    let quat = input.rotation;
    let theta: f64 = 2.0 * (quat.scalar()).acos();
    let half_theta = 0.5 * theta;
    let mut omega = na::Vector3::<f64>::zeros();

    let mut v_inv = na::Matrix3::<f64>::identity();
    if theta > 1e-6 {
        omega = quat.vector() * theta / (half_theta.sin());
        let ssym_omega = skew_sym(omega);
        v_inv -= ssym_omega * 0.5;
        v_inv += ssym_omega * ssym_omega * (1.0 - half_theta * half_theta.cos() / half_theta.sin())
            / (theta * theta);
    }

    let mut ret = na::Vector6::<f64>::zeros();
    ret.fixed_view_mut::<3, 1>(0, 0).copy_from(&(v_inv * t));
    ret.fixed_view_mut::<3, 1>(3, 0).copy_from(&omega);

    ret
}

/// Produces the Jacobian of the exponential map of a lie algebra transform
/// that is then applied to a point with respect to the transform.
///
/// i.e.
/// d exp(t) * p
/// ------------
///      d t
///
/// The parameter 'transformed_point' is assumed to be transformed already
/// and thus: transformed_point = exp(t) * p
///
/// This is largely taken from this paper:
/// https://ingmec.ual.es/~jlblanco/papers/jlblanco2010geometry3D_techrep.pdf
fn exp_map_jacobian(transformed_point: &na::Point3<f64>) -> na::Matrix3x6<f64> {
    let mut ss = na::Matrix3x6::zeros();
    ss.fixed_view_mut::<3, 3>(0, 0)
        .copy_from(&na::Matrix3::<f64>::identity());
    ss.fixed_view_mut::<3, 3>(0, 3)
        .copy_from(&(-skew_sym(transformed_point.coords)));
    ss
}

/// Projects a point in camera coordinates into the image plane
/// producing a floating-point pixel value
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

/// Jacobian of the projection function with respect to the 3D point in camera
/// coordinates. The 'transformed_pt' is a point already in
/// or transformed to camera coordinates.
fn proj_jacobian_wrt_point(
    intrinsic: &na::Matrix3<f64>,  //fx, fy, cx, cy
    distortion: &na::Vector4<f64>, //k1, k2, p1, p2
    transformed_pt: &na::Point3<f64>,
) -> na::Matrix2x3<f64> {
    let fx = intrinsic[(0, 0)];
    let fy = intrinsic[(1, 1)];
    let k1 = distortion[0];
    let k2 = distortion[1];
    let p1 = distortion[2];
    let p2 = distortion[3];

    let xn = transformed_pt.x / transformed_pt.z;
    let yn = transformed_pt.y / transformed_pt.z;
    let rn2 = xn * xn + yn * yn;
    let jacobian1 = na::Matrix2::<f64>::new(
        fx * (k1 * rn2 + k2 * rn2 * rn2 + 2.0 * p1 * yn + 4.0 * p2 * xn + 1.0),
        2.0 * fx * p1 * xn,
        2.0 * fy * p2 * yn,
        fy * (k1 * rn2 + k2 * rn2 * rn2 + 4.0 * p1 * yn + 2.0 * p2 * xn + 1.0),
    );
    let jacobian2 = na::Matrix2x3::<f64>::new(
        1.0 / transformed_pt.z,
        0.0,
        -transformed_pt.x / (transformed_pt.z.powi(2)),
        0.0,
        1.0 / transformed_pt.z,
        -transformed_pt.y / (transformed_pt.z.powi(2)),
    );
    jacobian1 * jacobian2
}

/// Struct for holding data for calibration.
struct Calibration<'a> {
    intrinsic: &'a na::Matrix3<f64>,
    distortion: &'a na::Vector4<f64>,
    model_pts: &'a Vec<na::Point3<f64>>,
    image_pts_set: &'a Vec<Vec<na::Point2<f64>>>,
    tfs: &'a Vec<na::Isometry3<f64>>,
}

impl<'a> Calibration<'a> {
    fn new(
        intrinsic: &'a na::Matrix3<f64>,
        distortion: &'a na::Vector4<f64>,
        model_pts: &'a Vec<na::Point3<f64>>,
        image_pts_set: &'a Vec<Vec<na::Point2<f64>>>,
        tfs: &'a Vec<na::Isometry3<f64>>,
    ) -> Self {
        Self {
            intrinsic, distortion, model_pts, image_pts_set, tfs,
        }
    }
    /// Decode the camera model and transforms from the flattened parameter vector
    ///
    /// The convention in use is that the first four values are the camera parameters
    /// and each set of six values after is a transform (one per image). This convention
    /// is also followed in the Jacobian function below.
    fn decode_params(
        &self,
        param: &na::DVector<f64>,
    ) -> Vec<na::Isometry3<f64>> {
        // Following the camera parameters, for each image there
        // will be one 6D transform
        let transforms = self
            .image_pts_set
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let lie_alg_transform: na::Vector6<f64> =
                    param.fixed_view::<6, 1>(6 * i, 0).clone_owned();
                // Convert to a useable na::Isometry
                exp_map(&lie_alg_transform)
            })
            .collect::<Vec<_>>();
        transforms
    }
}


impl lm::LMProblem for Calibration<'_> {
    fn residual(&self, param: &na::DVector<f64>) -> na::DVector<f64> {
                // Get usable camera model and transforms from the parameter vector
        let transforms = self.tfs;

        let num_images = self.image_pts_set.len();
        let num_target_points = self.model_pts.len();
        let num_residuals = num_images * num_target_points;

        // Allocate big empty residual
        let mut residual = na::DVector::<f64>::zeros(num_residuals * 2);

        let mut residual_idx = 0;
        for (idx, (image_pts, transform)) in self.image_pts_set.iter().zip(transforms.iter()).enumerate() {
            for (observed_image_pt, target_pt) in image_pts.iter().zip(self.model_pts.iter()) {
                // Apply image formation model
                let mut lie_algebra = log_map(transform);
                lie_algebra.x *= param[idx];
                lie_algebra.y *= param[idx];
                lie_algebra.z *= param[idx];

                let transformed_point = exp_map(&lie_algebra) * target_pt;
                let projected_pt = project(self.intrinsic, self.distortion, &transformed_point);

                // Populate residual vector two rows at time
                let individual_residual = projected_pt - observed_image_pt;
                residual
                    .fixed_view_mut::<2, 1>(residual_idx, 0)
                    .copy_from(&individual_residual);
                residual_idx += 2;
            }
        }

        residual
    }


    fn jacobian(&self, param: &na::DVector<f64>) -> na::DMatrix<f64> {
        // Get usable camera model and transforms from the parameter vector
        let transforms = self.tfs;

        let num_images = self.image_pts_set.len();
        let num_target_points = self.model_pts.len();
        let num_residuals = num_images * num_target_points;
        let num_unknowns = num_images;

        // Allocate big empty Jacobian
        let mut jacobian = na::DMatrix::<f64>::zeros(num_residuals * 2, num_unknowns);

        let mut residual_idx = 0;
        for (tform_idx, transform) in transforms.iter().enumerate() {
            for target_pt in self.model_pts.iter() {
                let mut lie_algebra = log_map(transform);
                lie_algebra.x *= param[tform_idx];
                lie_algebra.y *= param[tform_idx];
                lie_algebra.z *= param[tform_idx];
                // Apply image formation model
                let transformed_point = exp_map(&lie_algebra) * target_pt;

                // Populate the Jacobian part for the transform
                let proj_jacobian_wrt_point =
                    proj_jacobian_wrt_point(self.intrinsic, self.distortion, &transformed_point);
                let transform_jacobian_wrt_transform = exp_map_jacobian(&transformed_point);

                let lie_algebra = log_map(transform);
                let jacobian_xi_k = -na::Vector6::<f64>::new(
                    lie_algebra.x, lie_algebra.y, lie_algebra.z, 0.0, 0.0, 0.0
                );

                // Transforms come after camera parameters in sets of six columns
                jacobian
                    .fixed_view_mut::<2, 1>(residual_idx, tform_idx)
                    .copy_from(&(proj_jacobian_wrt_point * transform_jacobian_wrt_transform * jacobian_xi_k));

                residual_idx += 2;
            }
        }
        jacobian
    }
}

pub fn optimize_with_lm(
    k: &na::Matrix3<f64>, 
    dist: &na::Vector4<f64>, 
    tfs: &mut Vec<na::Isometry3<f64>>,
    img_points: &Vec<Vec<na::Point2<f64>>>,
    world_points: &[na::Point2<f64>]
) -> Result<(), Box<dyn std::error::Error>> {
    let world_points: Vec<_> = world_points.iter().map(|p| {
        na::Point3::<f64>::new(p.x, p.y, 0.0)
    }).collect();

    // Create calibration parameters
    let cal_cost = Calibration::new(k, dist, &world_points, img_points, tfs);

    let init_param = na::DVector::<f64>::identity(img_points.len());

    let max_iter = 100;
    let tol = 1e-8;
    let lambda = 1e-3;
    let gamma = 10.0;

    let res = crate::lm::levenberg_marquardt(cal_cost, init_param, max_iter, tol, lambda, gamma);

    for (idx, tf ) in tfs.iter_mut().enumerate() {
        let mut lie_algebra = log_map(tf);
        lie_algebra.x *= res[idx];
        lie_algebra.y *= res[idx];
        lie_algebra.z *= res[idx];
        *tf = exp_map(&lie_algebra);
    }

    eprintln!("{}\n\n", res);

    Ok(())
}


