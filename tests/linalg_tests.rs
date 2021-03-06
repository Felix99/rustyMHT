use filter::tracking_sim::Linalg;
use rm::linalg::Matrix;


#[test]
fn cholesky_decomposition() {
	let t = 2.0_f64;

	let q_matrix = Matrix::<f64>::new(4,4, vec![
            0.3 * t.powf(3.0), 0.0, 0.5 * t.powf(2.0), 0.0,
			0.0, 0.3 * t.powf(3.0), 0.0, 0.5 * t.powf(2.0),
			0.5 * t.powf(2.0), 0.0, t, 0.0,
			0.0, 0.5 * t.powf(2.0), 0.0, t]);
	
	let g = q_matrix.cholesky().unwrap();

	let result = &g * g.transpose() - q_matrix;
	for i in result.data() {
		assert!(i.abs() < 0.01);
	} 
}

#[test]
fn mvnrnd_draw_multivariate_numbers() {
	let mean = Matrix::<f64>::new(2,1,vec![2.0, 3.0]);
	let covar = Matrix::<f64>::new(2,2,vec![10.0, 4.0, 4.0, 7.0]);
	let mut la = Linalg::new();
	let n = 10000;
	let mut mean_samples = Matrix::<f64>::zeros(2,1);
	let mut covar_samples = Matrix::<f64>::zeros(2,2);
	let mut samples = Vec::new();
	for _ in 0..n {
		samples.push(la.mvnrnd(&mean,&covar));
	}

	for i in 0..n {
		mean_samples = mean_samples + &samples[i];		
	}
	mean_samples = mean_samples * (1_f64 / n as f64);
	
	for i in 0..n {
		let sample_vec = &samples[i] - &mean_samples;
		let spread_term = &sample_vec * &sample_vec.transpose();
		covar_samples = covar_samples + spread_term * (1_f64 / n as f64);
	}
	
	let result_mean = mean_samples - &mean;
	let result_covar = covar_samples - &covar;
	for i in result_mean.data() {
		assert!(i.abs() < 0.5);
	} 
	for i in result_covar.data() {
		assert!(i.abs() < 0.5);
	}
}


#[test]
pub fn inv() {
	let a = Matrix::<f64>::new(2,2,vec![10.0, 5.0, -3.0, 7.0]);
	let b = a.inverse().unwrap();
	let result = Matrix::<f64>::identity(2) - &a * &b;
	for i in result.data() {
		assert!(i.abs() < 0.01);
	}
}

#[test]
pub fn mvnpdf() {
    let x = Matrix::<f64>::new(2,1,vec![2.0, 5.0]);
    let m = Matrix::<f64>::new(2,1,vec![1.0, 3.0]);
    let c = Matrix::<f64>::new(2,2,vec![10.0, 3.0, 3.0, 8.0]);
    let la = Linalg::new();
    let res = la.normal(&x,&m,&c);
    assert!(res - 0.014658 < 1e-4);
}

#[test]
pub fn det() {
    let c = Matrix::<f64>::new(2,2,vec![10.0, 3.0, 3.0, 8.0]);
    let res = c.det();
    assert!(res - 71.0 < 1e-4);
}