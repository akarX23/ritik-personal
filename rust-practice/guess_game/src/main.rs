use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn main() {
    println!("Guess the number");

    let secret_number = rand::thread_rng().gen_range(1..=100);

    let mut guess: String = String::new();

    loop {
        println!("Please input your guess:");
        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read input!");
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_)=> continue
        };

        println!("Your guess is {guess}");

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small"),
            Ordering::Equal => {
                println!("Guessed correctly");
                break;
            }
            Ordering::Greater => println!("Too big"),
        }
    }
}
