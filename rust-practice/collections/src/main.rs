use std::io;
use std::collections::HashMap;

fn main() {
    pig_latin();
}

fn med_mod() {
    println!("Enter a list of numbers separated by spaces:");
    let mut list = String::new();

    io::stdin()
    .read_line(&mut list)
    .expect("Could not read list of numbers.");

    let mut num_list: Vec<usize> = Vec::new();

    for word in list.trim().split_whitespace() {
        match word.parse() {
            Ok(num) => num_list.push(num),
            Err(_) => println!("Parsing went wrong")
        }
    }

    num_list.sort();

    println!("The median is: {}", num_list[num_list.len() / 2]);

    let mut hm: HashMap<usize, usize> = HashMap::new();

    for num in num_list {
        *hm.entry(num).or_insert(0) += 1;
    }

    dbg!(&hm);
    println!("The mode is: {}", hm.keys().max_by_key(|x| hm[x]).unwrap());
}

fn pig_latin() {
    let vowels = String::from("aeiou");

    let mut input = String::new();

    io::stdin().read_line(&mut input).expect("Wrong input.");

    let mut output: Vec<String> = Vec::new();

    for word in input.split_whitespace() {
        let fir_char = &word.chars().next().unwrap();
        if vowels.contains(fir_char.to_ascii_lowercase()) {
            output.push(format!("{}-hay", word));
        } else {
            let rest = &word[fir_char.len_utf8()..];
            output.push(format!("{}-{}ay", rest, fir_char));
        }
    }

    dbg!(output);
}