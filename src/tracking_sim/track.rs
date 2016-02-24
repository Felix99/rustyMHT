use numeric::Tensor;

pub struct Track {
    pub state : Tensor<f64>,
    pub covar : Tensor<f64>,
    pub time : f64,
}