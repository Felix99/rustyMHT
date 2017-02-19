#![feature(std_misc, thread_sleep)]
extern crate rusty_machine as rm;
use rm::prelude::*;
use tracking_sim::{Config, Target, Linalg, Manager, Sensor, SimConfig, Track};
use plotter::Plotter;
use std::{thread, time};





pub fn run_sim() {
    let mut plotter = Plotter::new();
    //&plotter.test();
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
        plotter.clear();
        //plotter.plot_measurements(&msrs);
        let track = Track::new(&Matrix::zeros(2,1), &Matrix::zeros(2,2), 1.0);
        //let tracks = &manager.tracks.clone();

        let tracks = &manager.get_tracks();
        //plotter.plot_tracks(&manager.tracks);
        plotter.plot_tracks(&tracks);

        //let something = &manager.tracks.iter().map(|e| e.state.clone()).collect::<Vec<_>>();
        //plotter.plot_something(&something);
        for t in &manager.tracks {
            println!("{:?}", &t.state)
        }
        //plotter.set_fov(&sim_config);
        plotter.show();
        thread::sleep(time::Duration::from_millis(100));
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
