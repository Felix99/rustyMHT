use rusty_machine as rm;
use rm::prelude::*;
use gnuplot::{Figure, Caption, Color, PointSymbol, PointSize};
use tracking_sim::{Measurement, Track, SimConfig};
use gnuplot::{AxesCommon, AutoOption, Fix};
use std;

pub struct Plotter {
    figure: Figure,
}


impl Plotter {
    pub fn new() -> Plotter {
        Plotter {
            figure : Figure::new(),
        }
    }

    pub fn show(&mut self) {
        self.figure.show();
    }

    pub fn plot_measurements(&mut self, msrs: &Vec<Measurement>) {
        let x = msrs.iter().map(|e| &e.data[[0,0]]).collect::<Vec<_>>();
        let y = msrs.iter().map(|e| &e.data[[1,0]]).collect::<Vec<_>>();
        //println!("{:?}", y);
        &self.figure.axes2d()
            .points(x, y, &[Caption("Measurements"), Color("green"), PointSymbol('o')]);
        //&self.figure.show();
    }

    pub fn plot_tracks(&mut self, tracks: &Vec<Track>) {
        let x = &tracks.iter().map(|t| t.state[[0,0]].clone()).collect::<Vec<_>>();
        let x2 : Vec<f64> = x.iter().map(|e| e.clone()).collect::<Vec<_>>();
        let y = &tracks.iter().map(|t| t.state[[1,0]].clone()).collect::<Vec<_>>();
        let y2 : Vec<f64> = y.iter().map(|e| e.clone()).collect::<Vec<_>>();
        println!("{:?}", &y2);
        self.figure.axes2d()
            .points(x2, y2, &[Caption("Tracks"), Color("red"), PointSymbol('x')]);
    }

//    pub fn plot_something(&mut self, tracks: &Vec<Matrix<f64>>) {
//        let x = &tracks.iter().map(|e| e[[0,0]]).collect::<Vec<_>>();
//        let y = &tracks.iter().map(|e| e[[1,0]]).collect::<Vec<_>>();
//        print_type_of(&x);
//        &self.figure.axes2d()
//            .points(x, y, &[Caption("Tracks"), Color("red"), PointSymbol('x')]);
//    }


    pub fn set_fov(&mut self, config: &SimConfig) {
        self.figure.axes2d().set_x_ticks(Some((Fix(1.0), 1)), &[], &[])
            .set_x_range(Fix((config.fov.0).0), Fix((config.fov.0).1))
            .set_y_range(Fix((config.fov.1).0), Fix((config.fov.1).1));
    }

    pub fn clear(&mut self) {
        self.figure.clear_axes();
    }

}