#[macro_use(tensor)]
use filter::tracking_sim::Target;
use filter::tracking_sim::Dynamics;
use numeric::Tensor;


#[test]
fn main() {
    let initState = Tensor::<f64>::new(vec![10.0, 10.0, 2.0, 3.0]);
    let finalState = Tensor::<f64>::new(vec![15.0, 17.5, 2.0, 3.0]);
    let mut target = Target::new(initState,2.5,0.0);
    target.move_forward();
    assert!(&target.state == &finalState);
}


