use filter::tracking_sim::Target;
use filter::tracking_sim::Sensor;
use filter::tracking_sim::Filter;
use filter::tracking_sim::Config;
use filter::tracking_sim::Track;
use numeric::Tensor;


#[test]
fn Kalman_filter() {
    let filter = Filter::new();
    let init_state = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]);

}


