#![allow(dead_code)]
#![allow(non_snake_case)]

extern crate rusty_machine as rm;
extern crate rand;
extern crate matplotlib;
mod simulation;
mod tracking_sim;
mod plotter;
use simulation::run_sim;

fn main() {
	println!("Hello.");
	run_sim();
}
