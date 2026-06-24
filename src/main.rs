mod corpus;
mod drift;
mod hashing;
mod metrics;
mod obj_io;
mod perturb;
mod runner;

use std::process::ExitCode;

use clap::Parser;

/// p3d-spectral stability and discrimination bench.
#[derive(Parser, Debug, Clone)]
#[command(about, version)]
pub struct Config {
    /// Global multiplier on spectral bucket widths (calibration knob).
    #[arg(long, default_value_t = 1.0)]
    pub quant_scale: f64,

    /// Random seeds per perturbation instance.
    #[arg(long, default_value_t = 2)]
    pub seeds: u64,

    /// Quick mode: tiny corpus, single seed.
    #[arg(long)]
    pub quick: bool,

    /// Diagnostic mode: report per-dim spectral feature drift, no hashing.
    #[arg(long)]
    pub drift: bool,

    /// Output CSV path.
    #[arg(long, default_value = "results/results.csv")]
    pub out: String,
}

fn main() -> ExitCode {
    let cfg = Config::parse();
    let result = if cfg.drift {
        drift::run(&cfg)
    } else {
        runner::run(&cfg)
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
