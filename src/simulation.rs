extern crate rusty_machine as rm;
use rm::linalg::matrix::Matrix;

use tracking_sim::Config;

pub fn run_sim() {
    let config = Config {
            msr_covar : Matrix::<f64>::identity(2),
            msr_matrix : Matrix::<f64>::new(2,4,vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            q : 1.0,
            delta_t : 1.0,
            init_covar :  Matrix::<f64>::identity(4) * 100.0,
            p_D : 0.95,
            rho_F : 1e-6,
            mu_gating : 9.21,
            fov : None,
        };


}
