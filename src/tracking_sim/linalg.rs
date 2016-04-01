use rm::linalg::matrix::Matrix;
use rand::distributions::{Normal, Range, IndependentSample};
use rand::ThreadRng;
use rand;
use std::f64::consts::PI;

pub struct Linalg {
	i : i32,
    rs : ThreadRng,
}

impl Linalg {
	
	pub fn new() -> Linalg {
		Linalg {
			i : 1,
            rs : rand::thread_rng(),
		}
	}
	
	pub fn get(&self, A: &Matrix<f64>,i: usize, j: usize) -> f64 {
		A[[i,j]]
	}
	
	pub fn set(&self, A: &mut Matrix<f64>, i:usize, j:usize, val: f64) {
		let dim1 = A.cols();
		let mut data = A.mut_data();
		data[i * dim1 + j] = val;
	}
    
    
    pub fn copy(&self, A: &Matrix<f64>) -> Matrix<f64> {
        if A.cols() == 1 {      // Vector
            let mut B = Matrix::<f64>::zeros(A.rows(),1);
            let dim0 = A.rows();
            let data = A.data();
            for i in 0..dim0 {
                    self.set(&mut B,i,0,data[i]);
                }
            B
        } else {                // Matrix
            let mut B = Matrix::<f64>::zeros(A.rows(),A.cols());
            let dim0 = A.rows();
            let dim1 = A.cols();
            for i in 0..dim0 {
                for j in 0..dim1 {
                    self.set(&mut B,i,j,self.get(A,i,j));
                }
            }
            B
        }
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
		let L = covar.cholesky();
        let zero_mean_mvn = &L * stn;
        zero_mean_mvn + mean
	}

    pub fn stat_dist(&self, x: &Matrix<f64>, mean: &Matrix<f64>, covar: &Matrix<f64>) -> f64 {
        let covar_inv = covar.inverse();
        let x_cpy = self.copy(x);
        let d = &x_cpy - mean;
        let q = &d.transpose() * &covar_inv * &d;
        self.get(&q,0,0)
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
