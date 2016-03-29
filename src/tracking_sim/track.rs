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

    pub fn merge_hypotheses(&mut self) {
        let mut merged_hypos = Vec::new();
        let hypotheses_copy = self.hypotheses.iter().map(|h| h.copy()).collect();
        merged_hypos = self.merge_local(hypotheses_copy, merged_hypos);
        self.hypotheses = merged_hypos;
    }

    fn merge_local(&self, list_of_hypos: Vec<Hypothesis>, mut merged_hypos: Vec<Hypothesis>) -> Vec<Hypothesis> {
        if list_of_hypos.len() > 1 {
            let separated = self.find_elements_to_merge_with_tail(list_of_hypos);
            let mut to_merge = separated.0;
            let mut not_to_merge = separated.1;
            merged_hypos.push(self.merge_all(to_merge));
            self.merge_local(not_to_merge,merged_hypos)
        } else {
            merged_hypos
        }
    }

    fn find_elements_to_merge_with_tail(&self, mut list_of_hypos : Vec<Hypothesis>) -> (Vec<Hypothesis>,Vec<Hypothesis>) {
        let h_tail = list_of_hypos.pop().unwrap();
        let mut to_merge = vec![h_tail.copy()];
        let mut not_to_merge = Vec::new();
        for h in list_of_hypos.iter() {
            if self.hypotheses_are_close(&h,&h_tail) {
                to_merge.push(h.copy());
            } else {
                not_to_merge.push(h.copy());
            }
        }
        (to_merge,not_to_merge)
    }

    fn hypotheses_are_close(&self, h1: &Hypothesis, h2: &Hypothesis) -> bool {
        let d = &h1.state - &h2.state;
        let la = Linalg::new();
        let dist = la.get(&d.transpose().dot(&d),0,0);
        dist < 15.0
    }

    fn merge_all(&self, to_merge: Vec<Hypothesis>) -> Hypothesis {
        let weight = to_merge.iter().fold(0_f64, |a,e| a + e.weight);
        let mean = to_merge.iter().fold(Tensor::<f64>::zeros(&[self.dim,1]), |a,e| {
            let m = Tensor::eye(self.dim) * e.weight;
            a + &m.dot(&e.state)
        });
        let covar = to_merge.iter().fold(Tensor::<f64>::zeros(&[self.dim,self.dim]), |a,e| {
            let m = Tensor::eye(self.dim) * e.weight;
            let v = &e.state - &mean;
            let spread = v.dot(&v.transpose());
            let unweighted = &e.covar + &spread;
            a + &m.dot(&unweighted)
        });
        Hypothesis::new(&mean,&covar,weight)
    }
}

