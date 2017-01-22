extern crate rusty_machine as rm;
use rm::prelude::*;

use tracking_sim::Config;
use tracking_sim::Target;
use tracking_sim::Linalg;
use tracking_sim::Manager;
use tracking_sim::Sensor;


pub struct SimConfig {
    pub steps : u32,
    pub num_targets : u32,
    pub fov : ((f64,f64),(f64,f64)),
    pub config : Config,
}

impl SimConfig {
    pub fn new() -> SimConfig {
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

        SimConfig {
            steps : 100,
            num_targets : 10,
            fov : ((0.0,1e3),(0.0,1e3)),
            config : config,
        }
    }
}


pub fn run_sim() {
    let sim_config = SimConfig::new();
    let mut manager = Manager::new(sim_config.config.clone());
    let mut sensor = Sensor::new(sim_config.config.clone());
    let targets = generate_targets(&sim_config);
    for t in targets.iter() {
        println!("state: \n{}",t.state)
    }

    for step in 0..sim_config.steps {
        println!("step: {}", step);
        let msrs = sensor.measure(&targets);
        manager.process(msrs)

    }




}

fn generate_targets(config: &SimConfig) -> Vec<Target> {
    let mut la = Linalg::new();
    let x_size = (config.fov.0).1 - (config.fov.0).0;
    let y_size = (config.fov.1).1 - (config.fov.1).0;
    let mut targets = Vec::new();
    for _ in 0..config.num_targets {
        let x = la.uniform((config.fov.0).0,(config.fov.0).1);
        let y = la.uniform((config.fov.1).0,(config.fov.1).1);
        let vel = la.mvnrnd(&Matrix::<f64>::zeros(2,1),&(Matrix::<f64>::identity(2) * config.config.q));
        let state = Matrix::<f64>::new(4,1,vec![x,y,vel[[0,0]],vel[[1,0]]]);
        let new_target = Target::new(state,config.config.delta_t,config.config.q);
        targets.push(new_target);
    }
    targets
}
