extern crate rusty_machine as rm;
use rm::prelude::*;
use matplotlib::{Env, Plot};
use tracking_sim::{Measurement, Track, SimConfig};

pub struct Plotter<'a> {
    plot : Plot<'a>,
}


impl<'a> Plotter<'a> {

    pub fn new(env: &'a Env) -> Plotter<'a> {
        Plotter {
            plot : Plot::new(&env),
        }
    }

    pub fn show(&mut self) {
        self.plot.show();
    }

    pub fn plot_measurements(&mut self, msrs: &Vec<Measurement>) {
        let x = msrs.iter().map(|e| e.data[[0,0]].clone() as f32).collect::<Vec<_>>();
        let y = msrs.iter().map(|e| e.data[[1,0]].clone() as f32).collect::<Vec<_>>();
        self.plot
            .scatter(x.as_slice(), y.as_slice());
    }

    pub fn plot_tracks(&mut self, tracks: &Vec<Track>) {
        let x = tracks.iter().map(|e| e.state[[0,0]].clone() as f32).collect::<Vec<_>>();
        let y = tracks.iter().map(|e| e.state[[1,0]].clone() as f32).collect::<Vec<_>>();
        self.plot
            .scatter(x.as_slice(), y.as_slice());
    }

//    pub fn set_fov(&mut self, config: &SimConfig) {
//        self.figure.axes2d().set_x_ticks(Some((Fix(1.0), 1)), &[], &[])
//            .set_x_range(Fix((config.fov.0).0), Fix((config.fov.0).1))
//            .set_y_range(Fix((config.fov.1).0), Fix((config.fov.1).1));
//    }

    pub fn clear(&mut self) {
        self.plot.clf();
    }

}