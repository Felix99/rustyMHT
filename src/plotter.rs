extern crate rusty_machine as rm;
extern crate cpython;
use rm::prelude::*;
use tracking_sim::{Measurement, Track, SimConfig};

use cpython::Python;
use cpython::ObjectProtocol; //for call method
use cpython::{PyModule, PyTuple, PyDict};

pub struct Env {
    gil: cpython::GILGuard,
}

impl Env {
    pub fn new() -> Env {
        Env { gil: Python::acquire_gil() }
    }
}

pub struct Plotter<'a> {
    py: Python<'a>,
    plt: PyModule,
}

impl<'a> Plotter<'a> {

    pub fn new(env: &'a Env) -> Plotter<'a> {
        let py = env.gil.python();
        Plotter {
            py : py,
            plt : PyModule::import(py, "matplotlib.pyplot").unwrap()
        }
    }

    pub fn show(&mut self) {
        let _ = self.plt.call(self.py, "show", PyTuple::empty(self.py), None).unwrap();
    }

    pub fn plot_measurements(&mut self, msrs: &Vec<Measurement>) {
        let x = msrs.iter().map(|e| e.data[[0,0]].clone() as f32).collect::<Vec<_>>();
        let y = msrs.iter().map(|e| e.data[[1,0]].clone() as f32).collect::<Vec<_>>();
        let args = PyDict::new(self.py);
        &args.set_item(self.py, "marker", "o");
        &args.set_item(self.py, "c", "green");
        let _ = self.plt.call(self.py, "scatter", (x, y), Some(&args)).unwrap();
    }

    pub fn plot_tracks(&mut self, tracks: &Vec<Track>) {
        let x = tracks.iter().map(|e| e.state[[0,0]].clone() as f32).collect::<Vec<_>>();
        let y = tracks.iter().map(|e| e.state[[1,0]].clone() as f32).collect::<Vec<_>>();
        let args = PyDict::new(self.py);
        &args.set_item(self.py, "marker", "x");
        &args.set_item(self.py, "c", "red");
        let _ = self.plt.call(self.py, "scatter", (x, y), Some(&args)).unwrap();
    }

//    pub fn set_fov(&mut self, config: &SimConfig) {
//        self.figure.axes2d().set_x_ticks(Some((Fix(1.0), 1)), &[], &[])
//            .set_x_range(Fix((config.fov.0).0), Fix((config.fov.0).1))
//            .set_y_range(Fix((config.fov.1).0), Fix((config.fov.1).1));
//    }

    pub fn clear(&mut self) {

    }

}