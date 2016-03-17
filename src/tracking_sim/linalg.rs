#[macro_use(tensor)]
use numeric::Tensor;
use numeric::random::RandomState;
use std::f64::consts::PI;

pub struct Linalg {
	i : i32,
    rs : RandomState
}

impl Linalg {
	
	pub fn new() -> Linalg {
		Linalg {
			i : 1,
            rs : RandomState::new(1)
		}
	}
	
	pub fn get(&self, A: &Tensor<f64>,i: usize, j: usize) -> f64 {
		let dim1 = A.dim(1);
		let data = A.slice();
		data[i * dim1 + j]
	}
	
	pub fn set(&self, A: &mut Tensor<f64>, i:usize, j:usize, val: f64) {
		let dim1 = A.dim(1);
		let mut data = A.slice_mut();
		data[i * dim1 + j] = val;
	}

	pub fn scalar_matrix(q: f64, dim: usize) -> Tensor<f64> {
		Tensor::<f64>::eye(dim) * q
	}

    pub fn inv(&self, a: &Tensor<f64>) -> Tensor<f64> {
        assert!(a.dim(0) == 2);
        assert!(a.dim(1) == 2);
        let mut a_inv = Tensor::<f64>::zeros(&[2,2]);
        let a0 = self.get(a,0,0);
        let d0 = self.get(a,1,1);
        let b0 = self.get(a,0,1);
        let c0 = self.get(a,1,0);
        let det = a0 * d0 - b0 * c0;
        self.set(&mut a_inv, 0,0, d0 / det);
        self.set(&mut a_inv, 1,1, a0 / det);
        self.set(&mut a_inv, 0,1, -b0 / det);
        self.set(&mut a_inv, 1,0, -c0 / det);
        a_inv
    }

    pub fn det(&self, a: &Tensor<f64>) -> f64 {
        let a0 = self.get(a,0,0);
        let d0 = self.get(a,1,1);
        let b0 = self.get(a,0,1);
        let c0 = self.get(a,1,0);
        a0 * d0 - b0 * c0
    }

    pub fn copy(&self, A: &Tensor<f64>) -> Tensor<f64> {
        if A.ndim() == 1 {      // Vector
            let mut B = Tensor::<f64>::zeros(&[A.dim(0),1]);
            let dim0 = A.dim(0);
            let data = A.slice();
            for i in 0..dim0 {
                    self.set(&mut B,i,0,data[i]);
                }
            B
        } else {                // Matrix
            let mut B = Tensor::<f64>::zeros(&[A.dim(0),A.dim(1)]);
            let dim0 = A.dim(0);
            let dim1 = A.dim(1);
            for i in 0..dim0 {
                for j in 0..dim1 {
                    self.set(&mut B,i,j,self.get(A,i,j));
                }
            }
            B
        }
    }
	
	// Cholesky decomposition taken from https://de.wikipedia.org/wiki/Cholesky-Zerlegung
	pub fn cholesky (&self, B: &Tensor<f64>) -> Tensor<f64> {
		let mut A = self.copy(B);
		let dim = A.dim(1);
		assert!(A.dim(0) == A.dim(1));

		for i in 0..dim {
			for j in 0..i+1 {
				let mut sum = self.get(&A,i,j);
				for k in 0..j {
					sum = sum - self.get(&A,i,k) * self.get(&A,j,k);					
				}
				if i > j {
					let val = sum / self.get(&A,j,j);
					self.set(&mut A,i,j,val);
					} else if i == j {
						self.set(&mut A,i,i,sum.sqrt());
						if sum < 0.0 {
							panic!("Matrix not positive definite (numerically).");
						}
					}
			}
			for j in i+1..dim {
				self.set(&mut A,i,j,0.0);
			}
		}
		A
	}		
	
	pub fn mvnrnd(&mut self, mean: &Tensor<f64>, covar: &Tensor<f64>) -> Tensor<f64> {
		assert!(covar.dim(0) == covar.dim(1));
		let dim = covar.dim(0);
		let stn = self.rs.normal::<f64>(&[dim]);
		let L = self.cholesky(covar);
        let zero_mean_mvn = L.dot(&stn);
        zero_mean_mvn + mean
	}

    pub fn stat_dist(&self, x: &Tensor<f64>, mean: &Tensor<f64>, covar: &Tensor<f64>) -> f64 {
        let covar_inv = self.inv(covar);
        let x_cpy = self.copy(x);
        let d = &x_cpy - mean;
        let q = d.transpose().dot(&covar_inv).dot(&d);
        self.get(&q,0,0)
    }

    pub fn normal(&self, x: &Tensor<f64>, mean: &Tensor<f64>, covar: &Tensor<f64>) -> f64 {
        let det = self.det(covar);
        let c = 1_f64 / ((2_f64 * PI).powf(2_f64) * det).sqrt();
        let q = self.stat_dist(x,mean,covar);
        let e_q = (-1_f64 / 2_f64 * q).exp();
        c * e_q
    }

    pub fn uniform(&mut self, lower: f64, upper: f64) -> f64 {
        let z = self.rs.uniform::<f64>(lower, upper, &[1]);
        z.slice()[0]
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
