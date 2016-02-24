use numeric::Tensor;

pub struct Config {
    pub msr_covar: Tensor<f64>,
    pub msr_matrix: Tensor<f64>,
    pub q : f64,
    pub delta_t : f64,
    pub init_covar : Tensor<f64>,
}

impl Config {
    pub fn new() -> Config {
        Config {
            msr_covar : Tensor::<f64>::eye(2),
            msr_matrix : Tensor::<f64>::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).reshape(&[2,4]),
            q : 1.0,
            delta_t : 1.0,
            init_covar :  Tensor::<f64>::eye(2) * 100.0
        }
    }
}