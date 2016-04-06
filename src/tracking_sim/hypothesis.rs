use rm::linalg::matrix::Matrix;
use tracking_sim::Linalg;

pub struct Hypothesis {
    pub state : Matrix<f64>,
    pub covar : Matrix<f64>,
    pub weight : f64,
}

impl Hypothesis {
    pub fn new(state : &Matrix<f64>, covar: &Matrix<f64>, weight : f64) -> Hypothesis {
        Hypothesis {
            state : state.clone(),
            covar : covar.clone(),
            weight : weight,
        }
    }

}

impl Clone for Hypothesis {
    fn clone(&self) -> Hypothesis {
        Hypothesis::new(&self.state,&self.covar,self.weight)
    }
}