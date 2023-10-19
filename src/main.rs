#![feature(thread_id_value)]

use std::collections::HashMap;
use std::env;
use std::fs;
use std::os;
use std::io::prelude::*;
use std::path::Path;
use std::str::FromStr;
use std::io::LineWriter;

use crossbeam::channel::unbounded;
use rand::Rng;
use regex::Regex;
use reqwest::header::USER_AGENT;
use reqwest::Client;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use simple_pagerank::Pagerank;
use indicatif::ProgressBar;

const MAX_CONCURRENT: usize = 50;
const REQUEST_INTERVAL: u64 = 100; // ms
const HEADERS_LIST: [&'static str; 10] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.88 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36"
];
const ALPHA: f64 = 0.1;
const EPSILON: f64 = 0.000001;
const MAX_ITERATIONS: usize = 1000;

#[derive(Debug)]
struct URL {
    url: String,
    id: usize,
    crawled: bool,
}

impl URL {
    fn new<T, U>(url: T, id: U, crawled: bool) -> Self
    where
        T: Into<String>,
        U: Into<usize>,
    {
        URL {
            url: url.into(),
            id: id.into(),
            crawled,
        }
    }
}

#[cfg(feature = "pagerank")]
fn main() -> Result<(), std::io::Error> {
    pagerank()
}

#[tokio::main]
#[cfg(feature = "crawl")]
async fn main() {
    crawl().await
}

fn _pagerank() {
    let mut web_graph = Pagerank::<&str>::new();
    web_graph.set_damping_factor(((1f64 - ALPHA) * 100f64).floor() as u8);
    web_graph.add_edge("1", "2");
    web_graph.add_edge("3", "2");
    web_graph.add_edge("4", "2");
    web_graph.add_edge("4", "3");
    let _ = web_graph.calculate();
    println!("Page ranks: {:?}", web_graph.nodes())
}

fn pagerank() -> Result<(), std::io::Error> {
    let mut web_graph = Pagerank::<usize>::new();
    let _ = web_graph.set_damping_factor(((1f64 - ALPHA) * 100f64).floor() as u8).expect("Failed to set damping factor");

    let links_file = Path::new("res/url_list_short1.txt");
    let out_dir = Path::new("out");
    let regex = Regex::new("\"(https?://\\S+)\"").unwrap();

    // Read links.txt file
    let links_content = fs::read_to_string(links_file).expect("Failed to read links.txt");

    // Build link index using HashMap
    let link_index: HashMap<String, usize> = links_content
        .lines()
        .enumerate()
        .map(|(index, link)| (link.trim_end_matches('/').to_string(), index))
        .filter(|(_, index)| { fs::read_to_string(format!("{}.html", index)).unwrap_or(String::new()).len() == 0 })
        .collect();

    let bar = ProgressBar::new(link_index.len() as u64);

    // Process HTML files
    for (doc_id, _) in links_content.lines().enumerate() {
        bar.inc(1);

        // Read HTML file
        let html_file = out_dir.join(format!("{}.html", doc_id));
        let html_content = match fs::read_to_string(html_file) {
            Ok(content) => content,
            Err(_) => continue,
        };

        // Parse outgoing links from HTML content
        regex
            .captures_iter(html_content.as_str())
            .filter_map(|cap| {
                let link = cap.get(1).unwrap().as_str().trim_end_matches('/');
                link_index.get(link)
            })
            .map(|index| {
                    web_graph.add_edge(doc_id, *index);
            })
            .for_each(drop);
        // println!("{} Outgoing links: {:?}", doc_id, outgoing_link_indices);
    }

    let _ = web_graph.calculate();
    bar.finish();
    save_pagerank(web_graph.nodes(), links_content.lines().collect())
}

async fn crawl() {
    let urls = read_urls();
    let (tx, rx) = unbounded();
    for url in urls {
        tx.send(url).unwrap();
    }
    drop(tx);

    let client = Client::new();
    let mut handles = Vec::with_capacity(MAX_CONCURRENT);
    for tid in 0..MAX_CONCURRENT {
        let rx = rx.clone();
        let client = client.clone();
        handles.push(tokio::spawn(async move {
            while let Ok(url) = rx.recv() {
                // println!("Crawling {}", url.url);
                let filename = format!("out/{}.html", url.id);

                match download(&client, &url.url, &filename).await {
                    Ok(_) => println!("[{}] {} Success", tid, filename),
                    Err(err) => eprintln!("[{}] {} Failed {} : {}", tid, filename, url.url, err),
                }
            }
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }
}

fn save_pagerank(nodes: Vec<(&usize, f64)>, links: Vec<&str>) -> std::io::Result<()> {
    let file = fs::File::create("output/pagerank.txt")?;
    let mut file = LineWriter::new(file);
    for (doc_id, weight) in nodes {
        file.write_all(format!("[{}] {}: {}\n", doc_id, weight, links[*doc_id]).as_bytes())?;
    }
    Ok(())
}

fn read_urls() -> Vec<URL> {
    let args: Vec<String> = env::args().collect();
    let path = String::from_str("res/").unwrap() + &args[1];
    let out_dir = Path::new("out");
    println!("Reading from file {}", path);

    // 读取网址
    let mut urls = Vec::new();
    let mut f = fs::File::open(path).unwrap();
    let mut s = String::new();
    f.read_to_string(&mut s).unwrap();
    for (id, line) in s.lines().enumerate() {
        if !out_dir.join(format!("{}.html", id)).exists() {
            urls.push(URL::new(line, id, true));
        }
    }
    urls
}

async fn download(
    client: &Client,
    url: &str,
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let response = client
        .get(url)
        .header(USER_AGENT, get_random_header())
        .send()
        .await?;

    let mut file = File::create(filename).await?;
    file.write_all(response.text().await?.as_bytes()).await?;

    Ok(())
}

fn get_random_header() -> String {
    let mut rng = rand::thread_rng();
    let random_index = rng.gen_range(0..HEADERS_LIST.len());
    HEADERS_LIST[random_index].to_string()
}

