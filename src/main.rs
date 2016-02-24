#[macro_use(tensor)]

extern crate numeric;

mod simulation;
mod tracking_sim;
use simulation::run_sim;

fn main() {
	println!("Hello.");
	run_sim();
}
