#[macro_use(tensor)]

use numeric::Tensor;
use tracking_sim::Linalg;
use tracking_sim::Hypothesis;

pub struct Track {
    pub state : Tensor<f64>,
    pub covar : Tensor<f64>,
    pub time : f64,
    pub hypotheses : Vec<Hypothesis>,
    dim : usize,
}

impl Track {
    pub fn new(state : &Tensor<f64>, covar: &Tensor<f64>, time : f64) -> Track {
        let la = Linalg::new();
        Track {
            state : la.copy(state),
            covar : la.copy(covar),
            time : time,
            hypotheses : vec![Hypothesis::new(state,covar,1.0)],
            dim : state.dim(0),
        }
    }

    pub fn update_state(&mut self) {
        assert!(!self.hypotheses.is_empty());
        self.normalize_weights();
        self.state = self.hypotheses.iter().fold(Tensor::zeros(&[self.dim,1]), |s,e| {
            let m = Tensor::eye(self.dim) * e.weight;
            s + &m.dot(&e.state)
        });

        self.covar = self.hypotheses.iter().fold(Tensor::zeros(&[self.dim,self.dim]), |a,e| {
            let m = Tensor::eye(self.dim) * e.weight;
            let v = &e.state - &self.state;
            let spread = v.dot(&v.transpose());
            let unweighted = &e.covar + &spread;
            a + &m.dot(&unweighted)
        });

    }

    pub fn normalize_weights(&mut self) {
        let sum = self.hypotheses.iter().fold(0_f64, |a,e| a + e.weight);
        unsafe {
            for h in self.hypotheses.as_mut_slice() {
                h.weight = h.weight / sum;
            }
        }
    }
}

