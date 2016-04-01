use rm::linalg::matrix::Matrix;

use tracking_sim::Linalg;
use tracking_sim::Hypothesis;

pub struct Track {
    pub state : Matrix<f64>,
    pub covar : Matrix<f64>,
    pub time : f64,
    pub hypotheses : Vec<Hypothesis>,
    dim : usize,
}

impl Track {
    pub fn new(state : &Matrix<f64>, covar: &Matrix<f64>, time : f64) -> Track {
        let la = Linalg::new();
        Track {
            state : state.clone(),
            covar : covar.clone(),
            time : time,
            hypotheses : vec![Hypothesis::new(state,covar,1.0)],
            dim : state.rows(),
        }
    }

    pub fn update_state(&mut self) {
        assert!(!self.hypotheses.is_empty());
        self.normalize_weights();
        self.state = self.hypotheses.iter().fold(Matrix::zeros(self.dim,1), |s,e| {
            s + &e.state * e.weight
        });

        self.covar = self.hypotheses.iter().fold(Matrix::zeros(self.dim,self.dim), |a,e| {
            let v = &e.state - &self.state;
            let spread = &v * &v.transpose();
            let unweighted = &e.covar + spread;
            a + unweighted * e.weight
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
        let hypotheses_copy = self.hypotheses.clone();
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
        let mut to_merge = vec![h_tail.clone()];
        let mut not_to_merge = Vec::new();
        for h in list_of_hypos.iter() {
            if self.hypotheses_are_close(&h,&h_tail) {
                to_merge.push(h.clone());
            } else {
                not_to_merge.push(h.clone());
            }
        }
        (to_merge,not_to_merge)
    }

    fn hypotheses_are_close(&self, h1: &Hypothesis, h2: &Hypothesis) -> bool {
        let d = &h1.state - &h2.state;
        let metric = (&h1.covar + &h2.covar).inverse();
        let stat_dist = &d.transpose() * metric * &d;
        stat_dist[[0,0]]< 12.23
    }

    fn merge_all(&self, to_merge: Vec<Hypothesis>) -> Hypothesis {
        let weight = to_merge.iter().fold(0_f64, |a,e| a + e.weight);
        let mean = to_merge.iter().fold(Matrix::<f64>::zeros(self.dim,1), |a,e| {
            a + &e.state * e.weight
        });
        let covar = to_merge.iter().fold(Matrix::<f64>::zeros(self.dim,self.dim), |a,e| {
            let v = &e.state - &mean;
            let spread = &v * &v.transpose();
            let unweighted = &e.covar + spread;
            a + &unweighted * e.weight
        });
        Hypothesis::new(&mean,&covar,weight)
    }
}

impl Clone for Track {
    fn clone(&self) -> Track {
        Track {
            state : self.state.clone(),
            covar : self.covar.clone(),
            time : self.time,
            hypotheses : self.hypotheses.clone(),
            dim : self.dim,
        }
    }
}

