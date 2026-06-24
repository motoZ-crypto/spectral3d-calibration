//! Spectral identity wrapper: timing plus uniform outcome classification.
//!
//! The pipeline is pure math with input-proportional cost, so no timeout
//! machinery is needed — a failure is always an explicit `Err`.

use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct RegisteredIdentity {
    pub id: String,
    pub helper: spectral3d::Helper,
}

#[derive(Debug, Clone)]
pub enum RegisterOutcome {
    /// Registration accepted the shape and produced an identity plus helper.
    Ok(RegisteredIdentity),
    /// The registration shape gate intentionally rejected this geometry.
    Rejected(String),
    /// spectral3d rejected the mesh or parameters before identity creation.
    Error(String),
}

#[derive(Debug, Clone)]
pub enum VerifyOutcome {
    /// Recovered identity hash.
    Ok(String),
    /// spectral3d rejected the mesh or parameters during verification.
    Error(String),
}

fn params(quant_scale: f64) -> spectral3d::SpectralParams {
    let mut params = spectral3d::SpectralParams::default();
    params.quant.scale = quant_scale;
    params
}

/// Register OBJ bytes with the current spectral3d fuzzy-sketch pipeline.
pub fn register_obj(obj_bytes: &[u8], quant_scale: f64) -> (RegisterOutcome, Duration) {
    let started = Instant::now();
    let params = params(quant_scale);
    let outcome = match spectral3d::register(obj_bytes, &params) {
        Ok((id, helper)) => RegisterOutcome::Ok(RegisteredIdentity { id, helper }),
        Err(spectral3d::MeshError::WeakShape(e)) => RegisterOutcome::Rejected(e),
        Err(e) => RegisterOutcome::Error(format!("{e}")),
    };
    (outcome, started.elapsed())
}

/// Verify OBJ bytes against a registered helper, returning the recovered ID.
pub fn verify_obj(
    obj_bytes: &[u8],
    helper: &spectral3d::Helper,
    quant_scale: f64,
) -> (VerifyOutcome, Duration) {
    let started = Instant::now();
    let params = params(quant_scale);
    let outcome = match spectral3d::verify(obj_bytes, helper, &params) {
        Ok(id) => VerifyOutcome::Ok(id),
        Err(e) => VerifyOutcome::Error(format!("{e}")),
    };
    (outcome, started.elapsed())
}
