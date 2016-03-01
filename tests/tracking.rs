use filter::tracking_sim::Target;
use filter::tracking_sim::Sensor;
use filter::tracking_sim::Filter;
use filter::tracking_sim::Config;
use filter::tracking_sim::Track;
use filter::tracking_sim::Hypothesis;
use numeric::Tensor;


#[test]
fn Kalman_prediction() {
    let mut config = Config::new();
    config.p_D = 1.0;
    config.rho_F = 0.0;
    let mut filter = Filter::new();
    filter.set_config(config);
    let init_state = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]).reshape(&[4,1]);
    let config = Config::new();
    let mut track = Track::new(&init_state,&config.init_covar,1.0);
    filter.predict(&mut track);
    let final_state = Tensor::<f64>::new(vec![12.0, 13.0, 2.0, 3.0]).reshape(&[4,1]);
    let final_covar = Tensor::<f64>::new(vec![200.3, 0.0, 100.5, 0.0,
        0.0, 200.3, 0.0, 100.5,
        100.5, 0.0, 101.0, 0.0,
        0.0, 100.5, 0.0, 101.0]).reshape(&[4,4]);
    assert!(&track.state == &final_state);
    let diff_covar = &final_covar - &track.covar;
    for i in diff_covar.slice() {
        assert!(i < &0.01);
    }
}

#[test]
fn Hypotheses_merged() {
    let init_state1 = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]).reshape(&[4,1]);
    let init_state2 = Tensor::<f64>::new(vec![20.0, 25.0, -1.0, 2.0]).reshape(&[4,1]);
    let init_covar1 = Tensor::<f64>::eye(4) * 100.0;
    let init_covar2 = Tensor::<f64>::eye(4) * 75.0;
    let weight1 = 0.4;
    let weight2 = 0.6;
    let mut track = Track::new(&init_state1,&init_covar1,1.0);
    track.hypotheses = vec![Hypothesis::new(&init_state1,&init_covar1,weight1),
        Hypothesis::new(&init_state2,&init_covar2,weight2)];

    track.update_state();

    let final_state = Tensor::<f64>::new(vec![16.0, 19.0, 0.2, 2.4]).reshape(&[4,1]);
    let final_covar = Tensor::<f64>::new(vec![
    109.0000,   36.0000,   -7.2000,   -2.4000,
    36.0000,  139.0000,  -10.8000,   -3.6000,
    -7.2000,  -10.8000,   87.1600,    0.7200,
    -2.4000,   -3.6000,    0.7200,   85.2400]).reshape(&[4,4]);

    // compare state
    let diff_state = &final_state - &track.state;
    for i in diff_state.slice() {
        assert!(i < &0.01);
    }
    // compare covar
    let diff_covar = &final_covar - &track.covar;
    for i in diff_covar.slice() {
        assert!(i < &0.01);
    }
}


