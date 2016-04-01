extern crate rusty_machine as rm;
extern crate rand;

mod simulation;
mod tracking_sim;
use simulation::run_sim;

fn main() {
	println!("Hello.");
	run_sim();
}
