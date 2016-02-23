#[macro_use(tensor)]
use filter::tracking_sim::target::Target;
use filter::tracking_sim::sensor::Sensor;
use numeric::Tensor;


#[test]
fn measure_target_position() {
    let initState = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]);
    let target = Target::new(initState,2.5,0.0);
    let msr_covar = Tensor::<f64>::eye(2);
    let msr_matrix = Tensor::<f64>::new(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).reshape(&[2,4]);
    let mut sensor = Sensor::new(msr_covar,msr_matrix);

    let msr_covar = Tensor::<f64>::eye(2);

    let n = 10000;
    let scale_matrix = Tensor::<f64>::new(vec![1_f64/ n as f64, 0.0, 0.0, 1_f64 / n as f64]).reshape(&[2,2]);
    let mut samples = Vec::new();
    for _ in 0..n {
        samples.push(sensor.measure(&target).reshape(&[2,1]));
    }

    let mut mean_samples = Tensor::<f64>::zeros(&[2,1]);
    let mut covar_samples = Tensor::<f64>::zeros(&[2,2]);

    for i in 0..n {
    mean_samples = mean_samples + &samples[i];
    }
    mean_samples = scale_matrix.dot(&mean_samples);

    for i in 0..n {
    let sample_vec = &samples[i] - &mean_samples;
    let spread_term = sample_vec.dot(&sample_vec.transpose());
    covar_samples = covar_samples + spread_term.dot(&scale_matrix);
    }

    let result_mean = mean_samples - Tensor::<f64>::new(vec![10.0, 10.0]).reshape(&[2,1]);
    let result_covar = covar_samples - &msr_covar;
    for i in result_mean.slice() {
    assert!(i.abs() < 0.5);
    }
    for i in result_covar.slice() {
    assert!(i.abs() < 0.5);
    }
}


