use filter::tracking_sim::Linalg;
use numeric::Tensor;
#[macro_use(tensor)]

#[test]
fn matrix_copy() {
    let Q = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(&[2,4]);
    let la = Linalg::new();
    let B = la.copy(&Q);
    assert!(Q == B);

    let q = Tensor::<f64>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = la.copy(&q);
    let result = b - &q.reshape(&[8,1]);
    for i in result.slice() {
        assert!(i.abs() < 0.001);
    }
}

#[test]
fn cholesky_decomposition() {
	let T = 2.0_f64;

	let Q = Tensor::<f64>::new(vec![0.3 * T.powf(3.0), 0.0, 0.5 * T.powf(2.0), 0.0,
			0.0, 0.3 * T.powf(3.0), 0.0, 0.5 * T.powf(2.0),
			0.5 * T.powf(2.0), 0.0, T, 0.0,
			0.0, 0.5 * T.powf(2.0), 0.0, T]).reshape(&[4,4]);
	
	let la = Linalg::new();	

	let G = la.cholesky(&Q);
	let result = G.dot(&G.transpose()) - Q;
	for i in result.slice() {
		assert!(i.abs() < 0.01);
	} 
}

#[test]
fn mvnrnd_draw_multivariate_numbers() {
	let mean = Tensor::<f64>::new(vec![2.0, 3.0]);
	let covar = Tensor::<f64>::new(vec![10.0, 4.0, 4.0, 7.0]).reshape(&[2,2]);
	let mut la = Linalg::new();
	let N = 10000;
	let scale_matrix = Tensor::<f64>::new(vec![1_f64/ N as f64, 0.0, 0.0, 1_f64 / N as f64]).reshape(&[2,2]);
	let mut mean_samples = Tensor::<f64>::zeros(&[2,1]);
	let mut covar_samples = Tensor::<f64>::zeros(&[2,2]);
	let mut samples = Vec::new();
	for i in 0..N {
		samples.push(la.mvnrnd(&mean,&covar).reshape(&[2,1]));				
	}

	for i in 0..N {
		mean_samples = mean_samples + &samples[i];		
	}
	mean_samples = scale_matrix.dot(&mean_samples);
	
	for i in 0..N {		
		let sample_vec = &samples[i] - &mean_samples;
		let spread_term = sample_vec.dot(&sample_vec.transpose());
		covar_samples = covar_samples + spread_term.dot(&scale_matrix);
	}
	
	let result_mean = mean_samples - &mean.reshape(&[2,1]);
	let result_covar = covar_samples - &covar;
	for i in result_mean.slice() {
		assert!(i.abs() < 0.5);
	} 
	for i in result_covar.slice() {
		assert!(i.abs() < 0.5);
	}
}

#[test]
pub fn copy_matrix() {
	let la = Linalg::new();
	let A = Tensor::<f64>::new(vec![10.0, 4.0, -30.0, 7.0, 3.0, -2.0]).reshape(&[3,2]);
	let B = la.copy(&A);
	assert!(A==B);

}

#[test]
pub fn inv() {
	let la = Linalg::new();
	let A = Tensor::<f64>::new(vec![10.0, 5.0, -3.0, 7.0]).reshape(&[2,2]);
	let B = la.inv(&A);
	let result = &Tensor::<f64>::eye(2) - &A.dot(&B);
	for i in result.slice() {
		assert!(i.abs() < 0.01);
	}
}