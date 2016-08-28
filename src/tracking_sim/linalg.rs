//use rm::linalg::Matrix;
use rm::linalg::*;
use rand::distributions::{Normal, Range, IndependentSample};
use rand::ThreadRng;
use rand;
use std::f64::consts::PI;

pub struct Linalg {
    rs : ThreadRng,
}

impl Linalg {
	
	pub fn new() -> Linalg {
		Linalg {
            rs : rand::thread_rng(),
		}
	}
	
	pub fn set(&self, a: &mut Matrix<f64>, i:usize, j:usize, val: f64) {
		let dim1 = a.cols();
		let mut data = a.mut_data();
		data[i * dim1 + j] = val;
	}


    pub fn gen_std_normal_vec(&mut self, n: usize) -> Matrix<f64> {
        let mut b = Matrix::<f64>::zeros(n,1);
        {
            let normal = Normal::new(0.0,1.0);
            let mut data = b.mut_data();
            for i in 0..n {
                data[i] = normal.ind_sample(&mut self.rs);
            }
        }
        b
    }
	
	pub fn mvnrnd(&mut self, mean: &Matrix<f64>, covar: &Matrix<f64>) -> Matrix<f64> {
		assert!(covar.rows() == covar.cols());
		let dim = covar.rows();
		let stn = self.gen_std_normal_vec(dim);
		let l_res = covar.cholesky();
        if l_res.is_ok() {
            let zero_mean_mvn = &l_res.unwrap() * &stn;
            return zero_mean_mvn + mean
        } else {
            panic!("Cholesky decomposition failed!")
        }


	}

    pub fn stat_dist(&self, x: &Matrix<f64>, mean: &Matrix<f64>, covar: &Matrix<f64>) -> f64 {
        let covar_inv = covar.inverse();
        if covar_inv.is_ok() {
            let d = x - mean;
            return (&d.transpose() * &covar_inv.unwrap() * &d)[[0,0]]
        } else {
            panic!("Matrix inversion failed!")
        }


    }

    pub fn normal(&self, x: &Matrix<f64>, mean: &Matrix<f64>, covar: &Matrix<f64>) -> f64 {
        let det = covar.det();
        let c = 1_f64 / ((2_f64 * PI).powf(2_f64) * det).sqrt();
        let q = self.stat_dist(x,mean,covar);
        let e_q = (-1_f64 / 2_f64 * q).exp();
        c * e_q
    }

    pub fn uniform(&mut self, lower: f64, upper: f64) -> f64 {
        let uniform = Range::new(lower,upper);
        uniform.ind_sample(&mut self.rs)
    }

    pub fn poisson(&mut self,lambda : f64) -> i32 {
        let l = (-lambda).exp();
        let mut k = 0i32;
        let mut p = 1_f64;
        loop {
            k = k + 1i32;
            p = p * self.uniform(0_f64,1_f64);
            if p < l {break};
        }
        k-1
    }

}
