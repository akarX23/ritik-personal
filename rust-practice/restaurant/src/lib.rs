mod front_of_house;

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {
    hosting::add_to_waitlist();

    hosting::add_to_waitlist()
}

fn deliver_order() {}

mod back_of_house {
    fn fix_incorrect() {
        super::deliver_order();
    }
}