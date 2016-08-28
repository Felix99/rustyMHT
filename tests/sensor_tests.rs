use filter::tracking_sim::Target;
use filter::tracking_sim::Sensor;
use filter::tracking_sim::Config;
use rm::linalg::Matrix;


#[test]
fn measure_target_position() {
    let mut config = Config::new();
    let init_state = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let target = Target::new(init_state,2.5,0.0);
    let msr_covar = Matrix::<f64>::identity(2);
    config.msr_covar = msr_covar;
    config.p_D = 1.0;
    config.rho_F = 0.0;
    let mut sensor = Sensor::new(config);
    let msr_covar = Matrix::<f64>::identity(2);

    let n = 10000;
    let mut samples = Vec::new();
    for _ in 0..n {
        let z = sensor.measure(vec![&target]);
        samples.push(z);
    }

    let mut mean_samples = Matrix::<f64>::zeros(2,1);
    let mut covar_samples = Matrix::<f64>::zeros(2,2);

    for i in 0..n {
    mean_samples = &mean_samples + &samples[i].first().unwrap().data;
    }
    mean_samples = mean_samples * (1_f64 / n as f64);

    for i in 0..n {
    let sample_vec = &samples[i].first().unwrap().data - &mean_samples;
    let spread_term = &sample_vec * &sample_vec.transpose();
    covar_samples = covar_samples + spread_term * (1_f64 / n as f64);
    }

    let result_mean = mean_samples - Matrix::<f64>::new(2,1,vec![10.0, 10.0]);
    let result_covar = covar_samples - &msr_covar;
    for i in result_mean.data() {
    assert!(i.abs() < 0.5);
    }
    for i in result_covar.data() {
    assert!(i.abs() < 0.5);
    }
}

#[test]
fn false_measurements_a() {
    let mut config = Config::new();
    config.p_D = 0.0;
    config.rho_F = 1e-5;
    config.fov = Some(((-5000.0, 2500.0), (-4000.0, 2000.0)));
    let x_size = (&config.fov.unwrap().0).1 - (&config.fov.unwrap().0).0;
    let y_size = (&config.fov.unwrap().1).1 - (&config.fov.unwrap().1).0;
    let fov_size = x_size * y_size;
    let lambda = &fov_size * &config.rho_F; // lambda = 450
    let mut sensor = Sensor::new(config);
    let init_state = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let target = Target::new(init_state,2.5,0.0);
    let n = 250;

    // stats
    let mut num_msrs = Vec::new();

    for _ in 0..n {
        let msrs = sensor.measure(vec![&target]);
        num_msrs.push(msrs.len() as u64);
    }

    // compute statistics
    let sum = num_msrs.iter().fold(0, |a,e| a + e);
    let mean = sum as f64/ n as f64;

    let var = num_msrs.iter().fold(0_f64, |a,&e| a + (e as f64 - &mean).powf(2_f64) / n as f64);

    assert!(mean - lambda < 10.0);
    assert!(var - lambda < 100.0);
}


