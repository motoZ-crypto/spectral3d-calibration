//! Feature-drift diagnostics for the spectral pipeline.
//!
//! For every (perturbation kind, magnitude) this reports, per feature
//! dimension, the worst-case drift across the corpus measured in bucket
//! units (drift / QUANT_STEP). This is the instrument that turns bucket-
//! width tuning from guesswork into measurement:
//! - dims that must absorb a perturbation (expect=same) need drift < 0.5;
//! - dims that must detect one (expect=different) need drift > 0.5 in at
//!   least one discriminating dimension.

use std::collections::BTreeMap;

use spectral3d::{features::FEATURE_NAMES, N_FEATURES, QUANT_STEP};

use crate::corpus::{self, DEFAULT_RES};
use crate::obj_io::to_obj_bytes;
use crate::perturb;
use crate::Config;

/// Why an expect=same variant would lose its registered identity.
///
/// For every dimension whose plain bucket changes, report whether the
/// registered helper offset still recovers the base bucket (`~`) or whether
/// verification would lose the id (`!`).
fn flip_report(base: &[f64; N_FEATURES], var: &[f64; N_FEATURES], scale: f64) -> Vec<String> {
    let mut flips = Vec::new();
    for i in 0..N_FEATURES {
        // Mirror spectral3d::quant::{sketch,recover}.
        let xb = base[i] / (QUANT_STEP[i] * scale);
        let base_bucket = xb.round();
        let offset = xb - base_bucket;
        let xv = var[i] / (QUANT_STEP[i] * scale);
        if base_bucket as i64 != xv.round() as i64 {
            let recovered = (xv - offset).round();
            let rescuable = recovered as i64 == base_bucket as i64;
            flips.push(format!(
                "{}{}",
                FEATURE_NAMES[i],
                if rescuable { "~" } else { "!" }
            ));
        }
    }
    flips
}

fn spectral_features(obj: &[u8], target_samples: usize) -> Result<[f64; N_FEATURES], String> {
    let mesh = spectral3d::Mesh::parse_obj(obj).map_err(|e| e.to_string())?;
    let n = spectral3d::normalize(mesh).map_err(|e| e.to_string())?;
    let s = spectral3d::sample_surface(&n.mesh, target_samples);
    Ok(spectral3d::features(n.eigvals, &s))
}

pub fn run(cfg: &Config) -> Result<(), String> {
    let seeds = if cfg.quick { 1 } else { cfg.seeds };
    let shapes = corpus::corpus(cfg.quick);
    let target = spectral3d::SpectralParams::default().target_samples;

    // (kind, magnitude-as-string) -> per-dim max |drift| in bucket units
    let mut agg: BTreeMap<(String, String), [f64; N_FEATURES]> = BTreeMap::new();

    for spec in &shapes {
        let mesh = corpus::generate(spec, DEFAULT_RES.0, DEFAULT_RES.1);
        let base = spectral_features(&to_obj_bytes(&mesh), target)
            .map_err(|e| format!("{}: {e}", spec.name))?;
        for v in perturb::variants(spec, &mesh, seeds) {
            let f = match spectral_features(&to_obj_bytes(&v.mesh), target) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("  skip {} {}: {e}", spec.name, v.label);
                    continue;
                }
            };
            let key = (v.kind.to_string(), format!("{}", v.magnitude));
            let entry = agg.entry(key).or_insert([0.0; N_FEATURES]);
            for d in 0..N_FEATURES {
                let drift = (f[d] - base[d]).abs() / (QUANT_STEP[d] * cfg.quant_scale);
                if drift > entry[d] {
                    entry[d] = drift;
                }
            }
            let flips = flip_report(&base, &f, cfg.quant_scale);
            if !flips.is_empty() {
                println!(
                    "flips {:<22} {:<18} {}",
                    spec.name,
                    v.label,
                    flips.join(" ")
                );
            }
        }
        eprintln!("  drift: {} done", spec.name);
    }
    println!("(~ = plain bucket flipped but helper recovers it; ! = verification loses id)");

    println!(
        "\n=== max |feature drift| in bucket units (quant_scale={}) ===",
        cfg.quant_scale
    );
    println!("(expect=same rows want values < 0.5; stretch rows want some dim > 0.5)\n");
    print!("{:<10} {:>6}", "kind", "mag");
    for name in FEATURE_NAMES {
        print!(" {name:>6}");
    }
    println!();
    for ((kind, mag), drifts) in &agg {
        print!("{kind:<10} {mag:>6}");
        for d in drifts {
            print!(" {d:>6.2}");
        }
        println!();
    }
    Ok(())
}
