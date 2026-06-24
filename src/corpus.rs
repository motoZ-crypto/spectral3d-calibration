//! Deterministic corpus of procedurally generated test shapes.
//!
//! Every shape is a star-shaped radial function r(direction) sampled on a
//! lat-long grid: superellipsoid base + optional gaussian bumps. Star-shaped
//! guarantees simply-connected, closed, manifold meshes at any resolution,
//! which lets the remesh perturbation regenerate *the same geometry* with a
//! different tessellation - exactly what a different scanner would produce.
//!
//! Corpus groups:
//! - rocks:    asymmetric bumpy superellipsoids - the algorithm's intended diet
//! - bumpy:    bumpy spheres - weaker anisotropy, harder for PCA
//! - boxes:    aspect-ratio family (cube, stretched, plate) - regular-shape
//!   and PCA-degenerate traps
//! - nearsym:  near-symmetric shapes - PCA sign/order cliff triggers

use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, UnitSphere};

use crate::obj_io::TriMesh;

/// Gaussian bump on the direction sphere.
#[derive(Debug, Clone)]
pub struct Bump {
    pub dir: [f64; 3],
    /// Amplitude relative to base radius.
    pub amp: f64,
    /// Angular width in radians.
    pub sigma: f64,
}

/// Parametric star-shaped solid: superellipsoid + bumps.
#[derive(Debug, Clone)]
pub struct ShapeSpec {
    pub name: String,
    /// Semi-axes.
    pub a: f64,
    pub b: f64,
    pub c: f64,
    /// Superellipsoid exponent (2 = ellipsoid, 8 ~ rounded box).
    pub p: f64,
    pub bumps: Vec<Bump>,
}

impl ShapeSpec {
    /// Radius along unit direction `d`.
    pub fn radius(&self, d: [f64; 3]) -> f64 {
        let q = (d[0] / self.a).abs().powf(self.p)
            + (d[1] / self.b).abs().powf(self.p)
            + (d[2] / self.c).abs().powf(self.p);
        let mut r = q.powf(-1.0 / self.p);
        for bump in &self.bumps {
            let cosang =
                (d[0] * bump.dir[0] + d[1] * bump.dir[1] + d[2] * bump.dir[2]).clamp(-1.0, 1.0);
            let ang = cosang.acos();
            r += bump.amp * (-0.5 * (ang / bump.sigma).powi(2)).exp();
        }
        r
    }
}

/// Default mesh resolution: (rings, segments).
pub const DEFAULT_RES: (u32, u32) = (48, 96);

fn random_bumps(rng: &mut SmallRng, count: usize, amp_lo: f64, amp_hi: f64) -> Vec<Bump> {
    (0..count)
        .map(|_| Bump {
            dir: UnitSphere.sample(rng),
            amp: rng.random_range(amp_lo..amp_hi),
            sigma: rng.random_range(0.25..0.55),
        })
        .collect()
}

/// Build the deterministic corpus. `quick` keeps one shape per group.
pub fn corpus(quick: bool) -> Vec<ShapeSpec> {
    let mut shapes = Vec::new();

    // -- rocks: irregular solids spectral3d is supposed to handle well -------
    let rock_params: &[(f64, f64, f64, f64)] = &[
        (1.0, 0.75, 0.55, 2.2),
        (1.0, 0.85, 0.6, 1.8),
        (1.0, 0.7, 0.5, 2.8),
        (1.0, 0.8, 0.65, 2.0),
    ];
    for (i, &(a, b, c, p)) in rock_params.iter().enumerate() {
        let mut rng = SmallRng::seed_from_u64(1000 + i as u64);
        shapes.push(ShapeSpec {
            name: format!("rock{i}"),
            a,
            b,
            c,
            p,
            bumps: random_bumps(&mut rng, 8, 0.05, 0.12),
        });
    }

    // -- bumpy spheres --------------------------------------------------------
    for i in 0..3u64 {
        let mut rng = SmallRng::seed_from_u64(2000 + i);
        shapes.push(ShapeSpec {
            name: format!("bumpy{i}"),
            a: 1.0,
            b: 1.0,
            c: 1.0,
            p: 2.0,
            bumps: random_bumps(&mut rng, 10, 0.08, 0.15),
        });
    }

    // -- box family: the criticism's star witnesses ---------------------------
    let box_aspects: &[(&str, f64, f64, f64)] = &[
        ("box_cube", 1.0, 1.0, 1.0),
        ("box_112", 1.0, 1.0, 1.2),
        ("box_123", 1.0, 2.0, 3.0),
        ("box_133", 1.0, 3.0, 3.0),
    ];
    for &(name, a, b, c) in box_aspects {
        shapes.push(ShapeSpec {
            name: name.into(),
            a,
            b,
            c,
            p: 8.0,
            bumps: Vec::new(),
        });
    }

    // -- near-symmetric traps: PCA cliff triggers ------------------------------
    shapes.push(ShapeSpec {
        name: "nearsym_sphere_bump".into(),
        a: 1.0,
        b: 1.0,
        c: 1.0,
        p: 2.0,
        bumps: vec![Bump {
            dir: [1.0, 0.0, 0.0],
            amp: 0.03,
            sigma: 0.3,
        }],
    });
    shapes.push(ShapeSpec {
        name: "nearsym_ellipsoid".into(),
        a: 1.0,
        b: 1.02,
        c: 1.05,
        p: 2.0,
        bumps: Vec::new(),
    });

    if quick {
        let keep = ["rock0", "bumpy0", "box_cube", "nearsym_sphere_bump"];
        shapes.retain(|s| keep.contains(&s.name.as_str()));
    }
    shapes
}

/// Sample the spec on a lat-long grid: `rings` interior latitude rings of
/// `segments` vertices each, plus two pole vertices. Closed genus-0 manifold.
pub fn generate(spec: &ShapeSpec, rings: u32, segments: u32) -> TriMesh {
    assert!(rings >= 2 && segments >= 3);
    let r = rings as usize;
    let s = segments as usize;

    let mut vertices: Vec<[f64; 3]> = Vec::with_capacity(r * s + 2);

    let north = [0.0, 0.0, spec.radius([0.0, 0.0, 1.0])];
    vertices.push(north);

    for ri in 0..r {
        let theta = core::f64::consts::PI * (ri as f64 + 1.0) / (r as f64 + 1.0);
        let (st, ct) = (theta.sin(), theta.cos());
        for si in 0..s {
            let phi = core::f64::consts::TAU * si as f64 / s as f64;
            let d = [st * phi.cos(), st * phi.sin(), ct];
            let rad = spec.radius(d);
            vertices.push([rad * d[0], rad * d[1], rad * d[2]]);
        }
    }

    let south = [0.0, 0.0, -spec.radius([0.0, 0.0, -1.0])];
    vertices.push(south);
    let south_idx = (r * s + 1) as u32;

    let ring_start = |ri: usize| -> u32 { 1 + (ri * s) as u32 };

    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(2 * r * s);

    // North pole fan.
    for si in 0..s {
        let a = ring_start(0) + si as u32;
        let b = ring_start(0) + ((si + 1) % s) as u32;
        faces.push([0, a, b]);
    }
    // Quad strips between consecutive rings.
    for ri in 0..r - 1 {
        for si in 0..s {
            let a = ring_start(ri) + si as u32;
            let b = ring_start(ri) + ((si + 1) % s) as u32;
            let c = ring_start(ri + 1) + si as u32;
            let d = ring_start(ri + 1) + ((si + 1) % s) as u32;
            faces.push([a, c, d]);
            faces.push([a, d, b]);
        }
    }
    // South pole fan.
    for si in 0..s {
        let a = ring_start(r - 1) + si as u32;
        let b = ring_start(r - 1) + ((si + 1) % s) as u32;
        faces.push([a, south_idx, b]);
    }

    TriMesh { vertices, faces }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_is_deterministic() {
        let a = corpus(false);
        let b = corpus(false);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.name, y.name);
            assert_eq!(x.bumps.len(), y.bumps.len());
            for (bx, by) in x.bumps.iter().zip(y.bumps.iter()) {
                assert_eq!(bx.dir, by.dir);
                assert_eq!(bx.amp, by.amp);
            }
        }
    }

    #[test]
    fn mesh_is_closed_manifold_sized() {
        let spec = &corpus(true)[0];
        let m = generate(spec, 16, 32);
        assert_eq!(m.vertices.len(), 16 * 32 + 2);
        assert_eq!(m.faces.len(), 2 * 16 * 32);
        // Euler characteristic of a closed genus-0 mesh: V - E + F = 2.
        let v = m.vertices.len() as i64;
        let f = m.faces.len() as i64;
        let e = 3 * f / 2; // each edge shared by exactly 2 triangles
        assert_eq!(v - e + f, 2);
    }

    /// End-to-end: generated OBJ must be accepted by the spectral pipeline.
    #[test]
    fn spectral_accepts_generated_obj() {
        let spec = &corpus(true)[0];
        let mesh = generate(spec, 24, 48);
        let obj = crate::obj_io::to_obj_bytes(&mesh);
        let (id, helper) = spectral3d::register(&obj, &spectral3d::SpectralParams::default())
            .expect("spectral3d registration failed on generated OBJ");
        assert_eq!(id.len(), 64, "expected 64-char hex sha256");
        assert_eq!(helper.offsets.len(), spectral3d::N_FEATURES);
    }
}
