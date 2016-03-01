use numeric::Tensor;
use tracking_sim::Linalg;

pub struct Hypothesis {
    pub state : Tensor<f64>,
    pub covar : Tensor<f64>,
    pub weight : f64,
}

impl Hypothesis {
    pub fn new(state : &Tensor<f64>, covar: &Tensor<f64>, weight : f64) -> Hypothesis {
        let la = Linalg::new();
        Hypothesis {
            state : la.copy(state),
            covar : la.copy(covar),
            weight : weight,
        }
    }
}