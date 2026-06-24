//! Experiment orchestration: corpus x perturbations -> CSV + summary.

use std::io::Write;

use rayon::prelude::*;

use crate::corpus::{self, DEFAULT_RES};
use crate::hashing::{
    register_obj, verify_obj, RegisterOutcome, RegisteredIdentity, VerifyOutcome,
};
use crate::metrics::{self, Row};
use crate::obj_io::to_obj_bytes;
use crate::perturb;
use crate::Config;

fn register_fields(outcome: RegisterOutcome) -> (&'static str, Option<RegisteredIdentity>) {
    match outcome {
        RegisterOutcome::Ok(reg) => ("ok", Some(reg)),
        RegisterOutcome::Rejected(e) => {
            eprintln!("    rejected: {e}");
            ("rejected", None)
        }
        RegisterOutcome::Error(e) => {
            eprintln!("    error: {e}");
            ("error", None)
        }
    }
}

pub fn run(cfg: &Config) -> Result<(), String> {
    let seeds = if cfg.quick { 1 } else { cfg.seeds };
    let shapes = corpus::corpus(cfg.quick);

    eprintln!(
        "corpus: {} shapes | spectral quant_scale={} seeds={}",
        shapes.len(),
        cfg.quant_scale,
        seeds
    );

    // -- Phase 1: register all base shapes -----------------------------------
    let bases: Vec<_> = shapes
        .par_iter()
        .map(|spec| {
            let mesh = corpus::generate(spec, DEFAULT_RES.0, DEFAULT_RES.1);
            let obj = to_obj_bytes(&mesh);
            let (outcome, elapsed) = register_obj(&obj, cfg.quant_scale);
            let (status, registered) = register_fields(outcome);
            eprintln!(
                "  base {:<22} {:>8} {:>6}ms",
                spec.name,
                status,
                elapsed.as_millis(),
            );
            (spec.clone(), mesh, status.to_string(), registered, elapsed)
        })
        .collect();

    let mut rows: Vec<Row> = Vec::new();
    for (spec, _, status, _, elapsed) in &bases {
        rows.push(Row {
            shape: spec.name.clone(),
            kind: "base".into(),
            label: "base".into(),
            magnitude: 0.0,
            expectation: "-".into(),
            outcome: status.to_string(),
            elapsed_ms: elapsed.as_millis(),
            id_kept: None,
        });
    }

    // -- Phase 2: verify all variants -----------------------------------------
    struct Job {
        shape: String,
        base_id: String,
        helper: spectral3d::Helper,
        label: String,
        kind: &'static str,
        magnitude: f64,
        expectation: &'static str,
        obj: Vec<u8>,
    }

    let jobs: Vec<Job> = bases
        .iter()
        .flat_map(|(spec, mesh, status, registered, _)| {
            if status != "ok" {
                eprintln!("  skip variants of {} (base outcome: {status})", spec.name);
                return Vec::new();
            }
            let registered = registered
                .as_ref()
                .expect("ok base rows must carry a registered identity");
            perturb::variants(spec, mesh, seeds)
                .into_iter()
                .map(|v| Job {
                    shape: spec.name.clone(),
                    base_id: registered.id.clone(),
                    helper: registered.helper.clone(),
                    label: v.label,
                    kind: v.kind,
                    magnitude: v.magnitude,
                    expectation: v.expectation.as_str(),
                    obj: to_obj_bytes(&v.mesh),
                })
                .collect::<Vec<_>>()
        })
        .collect();

    eprintln!("variants: {} jobs", jobs.len());

    let variant_rows: Vec<Row> = jobs
        .into_par_iter()
        .map(|job| {
            let (outcome, elapsed) = verify_obj(&job.obj, &job.helper, cfg.quant_scale);
            let (status, kept) = match outcome {
                VerifyOutcome::Ok(id) => ("ok", Some(metrics::id_kept(&job.base_id, &id))),
                VerifyOutcome::Error(e) => {
                    eprintln!("    error: {e}");
                    ("error", None)
                }
            };
            eprintln!(
                "  {:<22} {:<16} {:>8} {:>6}ms id-kept={:?}",
                job.shape,
                job.label,
                status,
                elapsed.as_millis(),
                kept
            );
            Row {
                shape: job.shape,
                kind: job.kind.to_string(),
                label: job.label,
                magnitude: job.magnitude,
                expectation: job.expectation.to_string(),
                outcome: status.to_string(),
                elapsed_ms: elapsed.as_millis(),
                id_kept: kept,
            }
        })
        .collect();
    rows.extend(variant_rows);

    // -- Output ---------------------------------------------------------------
    let mut csv = String::with_capacity(rows.len() * 96);
    csv.push_str(metrics::CSV_HEADER);
    csv.push('\n');
    for r in &rows {
        csv.push_str(&r.to_csv());
        csv.push('\n');
    }
    let mut f = std::fs::File::create(&cfg.out).map_err(|e| format!("write {}: {e}", cfg.out))?;
    f.write_all(csv.as_bytes())
        .map_err(|e| format!("write {}: {e}", cfg.out))?;

    println!("\n=== stability / discrimination by (kind, magnitude) ===");
    println!("(id-kept: % of ok-runs where verification recovered the registered id;");
    println!(" expect=same wants 100%, expect=different wants 0%)\n");
    print!("{}", metrics::summarize(&rows));

    println!("\n=== cross-shape identity collisions (registered base vs base) ===");
    let base_ids: Vec<(String, String)> = bases
        .iter()
        .filter_map(|(spec, _, _, registered, _)| {
            registered
                .as_ref()
                .map(|reg| (spec.name.clone(), reg.id.clone()))
        })
        .collect();
    print!("{}", metrics::collision_report(&base_ids));

    println!("\nCSV written to {}", cfg.out);
    Ok(())
}
