#!/usr/bin/env rust-script
//! Benchmark: propagate the entire active Celestrak catalog to a single future time.
//!
//! This measures the "catalog sweep" pattern -- creating SGPPropagators for all
//! ~10,000 active satellites and stepping each to 1, 3, 5, and 7 days.
//! Both sequential and rayon-parallel modes are benchmarked.
//!
//! # Usage
//!
//! Default (auto-detected thread count, 10 iterations, both modes):
//!
//! ```sh
//! rust-script benchmarks/rust_catalog_bench.rs
//! ```
//!
//! Set rayon thread count via environment variable:
//!
//! ```sh
//! RAYON_NUM_THREADS=8 rust-script benchmarks/rust_catalog_bench.rs
//! ```
//!
//! Custom iterations:
//!
//! ```sh
//! rust-script benchmarks/rust_catalog_bench.rs --iterations 20
//! ```
//!
//! Run only the sequential or parallel benchmark:
//!
//! ```sh
//! rust-script benchmarks/rust_catalog_bench.rs --mode sequential
//! rust-script benchmarks/rust_catalog_bench.rs --mode parallel
//! ```
//!
//! Combined:
//!
//! ```sh
//! RAYON_NUM_THREADS=4 rust-script benchmarks/rust_catalog_bench.rs --iterations 5 --mode parallel
//! ```
//!
//! # Options
//!
//! | Flag             | Values                         | Default |
//! |------------------|--------------------------------|---------|
//! | `--iterations`   | Positive integer               | 10      |
//! | `--mode`         | `sequential`, `parallel`, `both` | `both`  |
//! | `RAYON_NUM_THREADS` | Env var: positive integer    | auto    |
//!
//! ```cargo
//! [dependencies]
//! brahe = "1.1"
//! rayon = "1"
//!
//! [profile.dev]
//! opt-level = 3
//! ```

use rayon::prelude::*;
use std::hint::black_box;
use std::panic::{self, AssertUnwindSafe};
use std::time::Instant;

use brahe::celestrak::CelestrakClient;
use brahe::eop::{EOPExtrapolation, FileEOPProvider};
use brahe::propagators::SGPPropagator;
use brahe::traits::SStatePropagator;

/// Duration scenarios: (label, tsince in seconds)
const DURATIONS: &[(&str, f64)] = &[
    ("1 day", 1.0 * 86400.0),
    ("3 days", 3.0 * 86400.0),
    ("5 days", 5.0 * 86400.0),
    ("7 days", 7.0 * 86400.0),
];

const DEFAULT_ITERATIONS: u32 = 10;

/// Simple argument parser for --iterations N and --mode MODE.
fn parse_args() -> (u32, String) {
    let args: Vec<String> = std::env::args().collect();
    let mut iterations = DEFAULT_ITERATIONS;
    let mut mode = "both".to_string();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--iterations" => {
                i += 1;
                if i < args.len() {
                    iterations = args[i]
                        .parse()
                        .expect("--iterations requires a positive integer");
                } else {
                    eprintln!("ERROR: --iterations requires a value");
                    std::process::exit(1);
                }
            }
            "--mode" => {
                i += 1;
                if i < args.len() {
                    mode = args[i].to_lowercase();
                    if !["sequential", "parallel", "both"].contains(&mode.as_str()) {
                        eprintln!(
                            "ERROR: --mode must be 'sequential', 'parallel', or 'both' (got '{}')",
                            mode
                        );
                        std::process::exit(1);
                    }
                } else {
                    eprintln!("ERROR: --mode requires a value");
                    std::process::exit(1);
                }
            }
            "--help" | "-h" => {
                println!("Usage: rust_catalog_bench.rs [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --iterations N   Timing iterations per scenario (default: {})", DEFAULT_ITERATIONS);
                println!("  --mode MODE      sequential, parallel, or both (default: both)");
                println!("  -h, --help       Show this help message");
                println!();
                println!("Environment:");
                println!("  RAYON_NUM_THREADS   Set the number of rayon threads (default: auto)");
                std::process::exit(0);
            }
            other => {
                eprintln!("ERROR: Unknown argument '{}'. Use --help for usage.", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    (iterations, mode)
}

fn run_sequential(satellites: &mut [SGPPropagator], n_sats: usize, iterations: u32) {
    println!("\n--- Sequential Propagation ---");

    let mut total_props_per_sec = 0.0;
    for (label, tsince_seconds) in DURATIONS {
        let mut total_ns = 0u128;
        let mut num_failed = 0u32;
        for _ in 0..iterations {
            let start = Instant::now();
            for prop in satellites.iter_mut() {
                if panic::catch_unwind(AssertUnwindSafe(|| {
                    black_box(prop.step_by(*tsince_seconds));
                }))
                .is_err()
                {
                    num_failed += 1;
                }
            }
            total_ns += start.elapsed().as_nanos();

            // Reset outside timing loop
            for prop in satellites.iter_mut() {
                prop.reset();
            }
        }

        let avg_ms = total_ns as f64 / iterations as f64 / 1_000_000.0;
        let props_per_sec = n_sats as f64 / (avg_ms / 1000.0);
        total_props_per_sec += props_per_sec;
        let failed_per_iter = num_failed / iterations;
        println!(
            "{:<30} {:>10.3} ms  ({:>12.2} prop/s, {} failed)",
            label, avg_ms, props_per_sec, failed_per_iter
        );
    }
    println!(
        "{:<30} {:>23.2} prop/s",
        "Average",
        total_props_per_sec / DURATIONS.len() as f64
    );
}

fn run_parallel(satellites: &mut [SGPPropagator], n_sats: usize, iterations: u32) {
    println!("\n--- Rayon Parallel Propagation ---");

    let mut total_props_per_sec = 0.0;
    for (label, tsince_seconds) in DURATIONS {
        let mut total_ns = 0u128;
        let mut num_failed = 0u32;
        for _ in 0..iterations {
            let start = Instant::now();
            let failed: u32 = satellites
                .par_iter_mut()
                .map(|prop| {
                    if panic::catch_unwind(AssertUnwindSafe(|| {
                        black_box(prop.step_by(*tsince_seconds));
                    }))
                    .is_err()
                    {
                        1u32
                    } else {
                        0u32
                    }
                })
                .sum();
            total_ns += start.elapsed().as_nanos();
            num_failed += failed;

            // Reset outside timing loop
            for prop in satellites.iter_mut() {
                prop.reset();
            }
        }

        let avg_ms = total_ns as f64 / iterations as f64 / 1_000_000.0;
        let props_per_sec = n_sats as f64 / (avg_ms / 1000.0);
        total_props_per_sec += props_per_sec;
        let failed_per_iter = num_failed / iterations;
        println!(
            "{:<30} {:>10.3} ms  ({:>12.2} prop/s, {} failed)",
            label, avg_ms, props_per_sec, failed_per_iter
        );
    }
    println!(
        "{:<30} {:>23.2} prop/s",
        "Average",
        total_props_per_sec / DURATIONS.len() as f64
    );
}

fn main() {
    let (iterations, mode) = parse_args();

    // Suppress default panic messages -- SGP4 panics on decayed orbits and we
    // catch those with catch_unwind; no need to flood stderr.
    panic::set_hook(Box::new(|_| {}));

    // --- Initialize EOP provider (required for time conversions / propagation) ---
    // Use FileEOPProvider instead of CachingEOPProvider (initialize_eop) because
    // CachingEOPProvider wraps its inner data in a Mutex, which serializes all
    // reads and destroys rayon parallelism. FileEOPProvider is immutable after
    // construction, so concurrent reads through the global RwLock are lock-free.
    let provider = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold)
        .expect("Failed to load bundled EOP data");
    brahe::eop::set_global_eop_provider(provider);

    // --- Download active catalog ---
    println!("Downloading active satellite catalog from Celestrak...");
    let client = CelestrakClient::new();
    let records = client
        .get_gp_by_group("active")
        .expect("Failed to download active catalog");
    println!("  Retrieved {} GP records", records.len());

    // --- Create propagators ---
    let mut satellites: Vec<SGPPropagator> = Vec::new();
    let mut failed = 0u32;
    for record in &records {
        match SGPPropagator::from_gp_record(record, 1.0) {
            Ok(prop) => satellites.push(prop),
            Err(_) => failed += 1,
        }
    }

    if satellites.is_empty() {
        eprintln!("ERROR: No satellites parsed successfully. Exiting.");
        std::process::exit(1);
    }

    let n_sats = satellites.len();

    println!("\nBrahe Catalog SGP4 Benchmark");
    println!("==================================================");
    println!("Rayon Threads: {}", rayon::current_num_threads());
    println!("Iterations: {}", iterations);
    println!(
        "Catalog: {} satellites ({} failed to parse)",
        n_sats, failed
    );

    // --- Warmup: 100 step_by calls on first satellite ---
    for _ in 0..100 {
        black_box(satellites[0].step_by(60.0));
        satellites[0].reset();
    }

    // --- Run requested benchmarks ---
    if mode == "sequential" || mode == "both" {
        run_sequential(&mut satellites, n_sats, iterations);
    }
    if mode == "parallel" || mode == "both" {
        run_parallel(&mut satellites, n_sats, iterations);
    }

    println!();
}
