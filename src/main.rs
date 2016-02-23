#[macro_use(tensor)]
extern crate numeric;

use numeric::random::RandomState;
use numeric::Tensor;

mod tracking_sim;
use tracking_sim::target::Target;
use tracking_sim::dynamics::Dynamics;
use tracking_sim::linalg::Linalg;

fn main() {
	let initState = tensor![0.0,0.0,1.0,0.5];
//	let dynamics = Dynamics::new(1.0,1.0);
	let mut target = Target::new(initState,2.0,1.0);

	
	println!("State: {}", target.state);
	target.move_forward();
	target.move_forward();

	println!("State: {}", target.state);	
	
	let mut x = tensor![1.0, 2.0];
	for i in 0..10 {
		x = x + tensor![1.0, 1.0];
	}
	
	println!("{}",Tensor::<f64>::zeros(&[2]));
	
	let mut rs = RandomState::new(1234);
	let x2 = tensor![1.0_f64, 2.0_f64].reshape(&[2,1]);
	let N = 1000;
	let y = x2 / N as f64;
	println!("{}",y);
	
	
	let mean = Tensor::<f64>::new(vec![2.0, 3.0]);
	let covar = Tensor::<f64>::new(vec![10.0, 4.0, 4.0, 7.0]).reshape(&[2,2]);
	let mut la = Linalg::new();
	let N = 1000; 
	let scale_matrix = Tensor::<f64>::new(vec![1_f64/ N as f64, 0.0, 0.0, 1_f64 / N as f64]).reshape(&[2,2]);
	let mut mean_samples = Tensor::<f64>::zeros(&[2,1]);
	let mut covar_samples = Tensor::<f64>::zeros(&[2,2]);
	let mut samples = Vec::new();
	for i in 0..N {
		samples.push(la.mvnrnd(&mean,&covar).reshape(&[2,1]));				
	}
	println!("first sample: {}", &samples[0]);
	for i in 0..N {
		mean_samples = mean_samples + &samples[i];		
	}
	mean_samples = scale_matrix.dot(&mean_samples);
	println!("mean samples: {}", &mean_samples);
	for i in 0..N {		
		let sample_vec = &samples[i];
		let spread_term = sample_vec.dot(&sample_vec.transpose()).dot(&scale_matrix);
		covar_samples = covar_samples + spread_term;
	}
	
	println!("{}\n{}",mean_samples,covar_samples);

	let x = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2,3]);
	println!("x: {}",x);
}
