extern crate rusty_machine as rm;
use rm::linalg::Matrix;

use tracking_sim::Config;
use tracking_sim::Target;
use tracking_sim::Linalg;

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
            fov : Some(((0.0,1e3),(0.0,1e3))),
            threshold_lr_upper : 1e3,
            threshold_lr_lower : 1e-3,
            threshold_pruning : 1e-2,
        };

    let num_targets = 10;
    let targets = generate_targets(&config,num_targets);
    for t in targets.iter() {
        println!("state: \n{}",t.state)
    }


}

fn generate_targets(config: &Config, num_targets : i32) -> Vec<Target> {
    let mut la = Linalg::new();
    let x_size = (config.fov.unwrap().0).1 - (config.fov.unwrap().0).0;
    let y_size = (config.fov.unwrap().1).1 - (config.fov.unwrap().1).0;
    let mut targets = Vec::new();
    for _ in 0..num_targets {
        let x = la.uniform((config.fov.unwrap().0).0,(config.fov.unwrap().0).1);
        let y = la.uniform((config.fov.unwrap().1).0,(config.fov.unwrap().1).1);
        let vel = la.mvnrnd(&Matrix::<f64>::zeros(2,1),&(Matrix::<f64>::identity(2) * config.q));
        let state = Matrix::<f64>::new(4,1,vec![x,y,vel[[0,0]],vel[[1,0]]]);
        let new_target = Target::new(state,config.delta_t,config.q);
        targets.push(new_target);
    }
    targets
}
