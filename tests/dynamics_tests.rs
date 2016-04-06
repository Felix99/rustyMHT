use rm::linalg::matrix::Matrix;
use filter::tracking_sim::Target;




#[test]
fn main() {
    let init_state = Matrix::<f64>::new(4,1,vec![10.0, 10.0, 2.0, 3.0]);
    let final_state = Matrix::<f64>::new(4,1,vec![15.0, 17.5, 2.0, 3.0]);
    let mut target = Target::new(init_state,2.5,0.0);
    target.move_forward();
    let res = &target.state - &final_state;
    for i in res.data() {
        assert!(i.abs() < 1e-3);
    }
}


