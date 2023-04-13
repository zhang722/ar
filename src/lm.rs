use nalgebra as na;
use na::{DMatrix, DVector};

pub trait LMProblem {
    fn residual(&self, p: &DVector<f64>) -> DVector<f64>;
    fn jacobian(&self, p: &DVector<f64>) -> DMatrix<f64>;
}

pub fn levenberg_marquardt<P>(
    problem: P,
    init_param: DVector<f64>,
    max_iter: usize,
    tol: f64,
    lambda: f64,
    gamma: f64,
) -> DVector<f64>
where
    P: LMProblem,
{
    let mut x = init_param;
    let mut lambda_current = lambda;
    let mut iteration = 0;

    while iteration < max_iter {
        let fx = problem.residual(&x);
        let j = problem.jacobian(&x);

        let jtj = j.transpose() * &j;
        let mut jtj_with_lambda = jtj.clone();
        for i in 0..jtj_with_lambda.nrows() {
            jtj_with_lambda[(i, i)] += lambda_current;
        }

        let delta_x = jtj_with_lambda.lu().solve(&(j.transpose() * -fx.clone()));

        match delta_x {
            Some(dx) => {
                let new_x = &x + &dx;

                let fx_new = problem.residual(&new_x);
                let fx_norm = fx.norm();
                let fx_new_norm = fx_new.norm();

                if (fx_norm - fx_new_norm).abs() < tol {
                    break;
                }

                if fx_new_norm < fx_norm {
                    lambda_current /= gamma;
                    x = new_x;
                } else {
                    lambda_current *= gamma;
                }
            }
            None => {
                lambda_current *= gamma;
            }
        }

        iteration += 1;
    }

    x
}
