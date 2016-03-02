use numeric::Tensor;

pub struct Measurement {
    pub data : Tensor<f64>,
}

impl Measurement {
    pub fn new(msrs: Vec<f64>) -> Measurement {
        let dim = msrs.len() as isize;
        Measurement {
            data : Tensor::<f64>::new(msrs).reshape(&[dim,1]),
        }
    }
}

