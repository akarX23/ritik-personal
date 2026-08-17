fn main() {
    
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
        };
        line_ele += 1;
        println!("");
    }
}

