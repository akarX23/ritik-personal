fn main() {
    println!("Result is: {}", add(5, 6));

    loop_test();

    println!("Far To Cel: {}", far_to_cel(97.0));

    fibo_n(15);

    let mut temp_str = String::from("world, ");
    ref_me(&mut temp_str);
    println!("The string: {}", temp_str);

    let rect1 = Rectangle {
        height: 32.5,
        width: 10.5,
    };
    let rect2: Rectangle = Rectangle {
        height: 16.0,
        width: 16.0,
    };
    println!("Can rect1 hold rec2: {}", rect1.can_hold(&rect2));
    println!("Area: {}", rect1.area());
    dbg!(&rect1);

    enum_play();
}

#[derive(Debug)]
struct Rectangle {
    height: f32,
    width: f32,
}

impl Rectangle {
    fn area(&self) -> f32 {
        self.height * self.width
    }
    fn can_hold(&self, ref_rect: &Rectangle) -> bool {
        self.width > ref_rect.width && self.height > ref_rect.height
    }
}

#[derive(Debug)]
struct IpAddr4 {
    n_octets: i32,
    value: String,
}

#[derive(Debug)]
struct IpAddr6 {
    n_octets: i32,
    value: String,
}

#[derive(Debug)]
enum IpAddr {
    V4(IpAddr4),
    V6(IpAddr6),
}

fn add(a: usize, b: usize) -> usize {
    a + b
}

fn loop_test() {
    let mut counter = 0;

    let result = loop {
        counter += 1;

        if counter > 5 {
            break counter * 2;
        }
    };

    println!("Result is: {result}");
}

fn far_to_cel(far: f32) -> f32 {
    return (far - 32.0) * (5.0 / 9.0);
}

fn fibo_n(n: usize) {
    let mut curr: usize = 0;
    let mut next: usize = 1;
    let mut line_ele: usize = 1;

    let mut n = n - 2;

    while n > 0 {
        for _ in 1..line_ele {
            print!("{curr} ");
            let temp: usize = curr;
            curr = next;
            next = next + temp;
            n = if n > 0 { n - 1 } else { 0 };
        }
        line_ele += 1;
        println!("");
    }
}

fn ref_me(temp_str_ref: &mut String) {
    temp_str_ref.push_str("hello");
}

fn enum_play() {
    let home: IpAddr = IpAddr::V4(IpAddr4 {
        n_octets: 4,
        value: String::from("127.0.0.1"),
    });
    let loopback: IpAddr = IpAddr::V6(IpAddr6 {
        n_octets: 6,
        value: String::from("::1"),
    });
    
    println!("{home:#?}{loopback:#?}");
    match home {
        IpAddr::V4(address) => {
            println!("IPv{} address: {}", address.n_octets, address.value);
        }
        IpAddr::V6(address) => {
            println!("IPv{} address: {}", address.n_octets, address.value);
        }
    }
}
