//! Mesh perturbation operators - one per vision promise under test.
//!
//! | kind    | simulates                          | vision expectation        |
//! |---------|------------------------------------|---------------------------|
//! | rigid   | object in a different pose         | same identity             |
//! | reorder | scanner emitting same geometry in  | same identity (pure       |
//! |         | different vertex/face order        | implementation determinism)|
//! | remesh  | scanner with different sampling    | same identity             |
//! | scale   | unit convention change (mm vs cm)  | same identity (calibrated)|
//! | noise   | scan noise on the surface          | same identity             |
//! | stretch | a genuinely different object       | DIFFERENT identity        |
//! | tamper  | local modification (forgery probe) | gray zone by construction |

use std::hash::{Hash, Hasher};

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, StandardNormal, UnitSphere};

use crate::corpus::{generate, ShapeSpec};
use crate::obj_io::TriMesh;

/// Derive a deterministic seed from a base value and string labels, so each
/// (shape, operator, instance) gets its own reproducible stream.
fn seed(base: u64, labels: &[&str]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    base.hash(&mut h);
    for l in labels {
        l.hash(&mut h);
    }
    h.finish()
}

/// What the vision promises for this variant relative to the base shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expectation {
    /// Must produce the same identity (stability requirement).
    Same,
    /// Must produce a different identity (discrimination requirement).
    Different,
    /// Boundary case - reported as a curve, not pass/fail.
    Gray,
}

impl Expectation {
    pub fn as_str(self) -> &'static str {
        match self {
            Expectation::Same => "same",
            Expectation::Different => "different",
            Expectation::Gray => "gray",
        }
    }
}

/// A perturbed instance of a base shape, ready to hash.
pub struct Variant {
    /// e.g. "noise_0.005_s1"
    pub label: String,
    /// Aggregation group: rigid|reorder|remesh|scale|noise|stretch|tamper.
    pub kind: &'static str,
    /// Numeric knob for stability curves (0.0 where not applicable).
    pub magnitude: f64,
    pub expectation: Expectation,
    pub mesh: TriMesh,
}

/// Build the full variant suite for one base shape.
///
/// `seeds` controls how many independent random instances are produced for
/// the stochastic operators (rigid, reorder, noise, tamper).
pub fn variants(spec: &ShapeSpec, base: &TriMesh, seeds: u64) -> Vec<Variant> {
    let mut out = Vec::new();
    let rms = base.rms_radius();

    // Control: the untouched mesh, re-serialized and verified. Anything below
    // 100% id-kept here means the bench itself (or the algorithm) is
    // non-deterministic.
    out.push(Variant {
        label: "control_identity".into(),
        kind: "control",
        magnitude: 0.0,
        expectation: Expectation::Same,
        mesh: base.clone(),
    });

    for s in 0..seeds {
        let label = format!("rigid{s}");
        let mut rng = SmallRng::seed_from_u64(seed(0xA11CE, &[spec.name.as_str(), label.as_str()]));
        out.push(Variant {
            label: format!("rigid_s{s}"),
            kind: "rigid",
            magnitude: 0.0,
            expectation: Expectation::Same,
            mesh: rigid(base, &mut rng),
        });
    }

    for s in 0..seeds {
        let label = format!("reorder{s}");
        let mut rng = SmallRng::seed_from_u64(seed(0xA11CE, &[spec.name.as_str(), label.as_str()]));
        out.push(Variant {
            label: format!("reorder_s{s}"),
            kind: "reorder",
            magnitude: 0.0,
            expectation: Expectation::Same,
            mesh: reorder(base, &mut rng),
        });
    }

    for &(rings, segs) in &[(32u32, 64u32), (64u32, 128u32)] {
        out.push(Variant {
            label: format!("remesh_{rings}x{segs}"),
            kind: "remesh",
            magnitude: rings as f64,
            expectation: Expectation::Same,
            mesh: generate(spec, rings, segs),
        });
    }

    for &k in &[0.5f64, 2.0] {
        out.push(Variant {
            label: format!("scale_{k}"),
            kind: "scale",
            magnitude: k,
            expectation: Expectation::Same,
            mesh: scale(base, k),
        });
    }

    for &sigma in &[0.001f64, 0.005, 0.01] {
        for s in 0..seeds {
            let label = format!("noise{sigma}{s}");
            let mut rng = SmallRng::seed_from_u64(seed(0xA11CE, &[spec.name.as_str(), label.as_str()]));
            out.push(Variant {
                label: format!("noise_{sigma}_s{s}"),
                kind: "noise",
                magnitude: sigma,
                expectation: Expectation::Same,
                mesh: radial_noise(base, sigma * rms, &mut rng),
            });
        }
    }

    for &k in &[1.05f64, 1.2] {
        out.push(Variant {
            label: format!("stretch_z{k}"),
            kind: "stretch",
            magnitude: k,
            expectation: Expectation::Different,
            mesh: stretch_z(base, k),
        });
    }

    for &amp in &[0.01f64, 0.05] {
        for s in 0..seeds {
            let label = format!("tamper{amp}{s}");
            let mut rng = SmallRng::seed_from_u64(seed(0xA11CE, &[spec.name.as_str(), label.as_str()]));
            out.push(Variant {
                label: format!("tamper_{amp}_s{s}"),
                kind: "tamper",
                magnitude: amp,
                expectation: Expectation::Gray,
                mesh: tamper_bump(base, amp * rms, 0.3, &mut rng),
            });
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

/// Random rotation (axis-angle) + translation of up to half the RMS radius.
fn rigid(mesh: &TriMesh, rng: &mut SmallRng) -> TriMesh {
    let axis: [f64; 3] = UnitSphere.sample(rng);
    let angle = rng.random_range(0.0..core::f64::consts::TAU);
    let t = [
        rng.random_range(-0.5..0.5) * mesh.rms_radius(),
        rng.random_range(-0.5..0.5) * mesh.rms_radius(),
        rng.random_range(-0.5..0.5) * mesh.rms_radius(),
    ];
    let r = rotation_matrix(axis, angle);
    let vertices = mesh
        .vertices
        .iter()
        .map(|v| {
            [
                r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2] + t[0],
                r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2] + t[1],
                r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2] + t[2],
            ]
        })
        .collect();
    TriMesh {
        vertices,
        faces: mesh.faces.clone(),
    }
}

/// Rodrigues rotation matrix.
fn rotation_matrix(axis: [f64; 3], angle: f64) -> [[f64; 3]; 3] {
    let (x, y, z) = (axis[0], axis[1], axis[2]);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}

/// Permute vertex order (remapping faces) and shuffle face order.
/// Geometry is bit-identical; only the file layout changes.
fn reorder(mesh: &TriMesh, rng: &mut SmallRng) -> TriMesh {
    let n = mesh.vertices.len();
    let mut perm: Vec<usize> = (0..n).collect(); // perm[new_index] = old_index
    perm.shuffle(rng);
    let mut inverse = vec![0u32; n];
    for (new_i, &old_i) in perm.iter().enumerate() {
        inverse[old_i] = new_i as u32;
    }
    let vertices: Vec<[f64; 3]> = perm.iter().map(|&old| mesh.vertices[old]).collect();
    let mut faces: Vec<[u32; 3]> = mesh
        .faces
        .iter()
        .map(|f| {
            [
                inverse[f[0] as usize],
                inverse[f[1] as usize],
                inverse[f[2] as usize],
            ]
        })
        .collect();
    let mut fperm: Vec<usize> = (0..faces.len()).collect();
    fperm.shuffle(rng);
    faces = fperm.iter().map(|&i| faces[i]).collect();
    TriMesh { vertices, faces }
}

/// Uniform scale about the origin.
fn scale(mesh: &TriMesh, k: f64) -> TriMesh {
    TriMesh {
        vertices: mesh
            .vertices
            .iter()
            .map(|v| [v[0] * k, v[1] * k, v[2] * k])
            .collect(),
        faces: mesh.faces.clone(),
    }
}

/// Anisotropic stretch along z - a genuinely different shape.
fn stretch_z(mesh: &TriMesh, k: f64) -> TriMesh {
    TriMesh {
        vertices: mesh
            .vertices
            .iter()
            .map(|v| [v[0], v[1], v[2] * k])
            .collect(),
        faces: mesh.faces.clone(),
    }
}

/// Independent radial displacement per vertex: v += dir(v) * N(0, sigma_abs).
fn radial_noise(mesh: &TriMesh, sigma_abs: f64, rng: &mut SmallRng) -> TriMesh {
    let c = mesh.centroid();
    let vertices = mesh
        .vertices
        .iter()
        .map(|v| {
            let mut d = [v[0] - c[0], v[1] - c[1], v[2] - c[2]];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if len > 1e-12 {
                d = [d[0] / len, d[1] / len, d[2] / len];
            } else {
                d = [0.0, 0.0, 1.0];
            }
            let z: f64 = StandardNormal.sample(rng);
            let off = sigma_abs * z;
            [v[0] + d[0] * off, v[1] + d[1] * off, v[2] + d[2] * off]
        })
        .collect();
    TriMesh {
        vertices,
        faces: mesh.faces.clone(),
    }
}

/// Local gaussian bump at a random direction: the "forgery probe".
fn tamper_bump(mesh: &TriMesh, amp_abs: f64, sigma_ang: f64, rng: &mut SmallRng) -> TriMesh {
    let c = mesh.centroid();
    let bump_dir: [f64; 3] = UnitSphere.sample(rng);
    let vertices = mesh
        .vertices
        .iter()
        .map(|v| {
            let mut d = [v[0] - c[0], v[1] - c[1], v[2] - c[2]];
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if len > 1e-12 {
                d = [d[0] / len, d[1] / len, d[2] / len];
            } else {
                d = [0.0, 0.0, 1.0];
            }
            let cosang =
                (d[0] * bump_dir[0] + d[1] * bump_dir[1] + d[2] * bump_dir[2]).clamp(-1.0, 1.0);
            let ang = cosang.acos();
            let off = amp_abs * (-0.5 * (ang / sigma_ang).powi(2)).exp();
            [v[0] + d[0] * off, v[1] + d[1] * off, v[2] + d[2] * off]
        })
        .collect();
    TriMesh {
        vertices,
        faces: mesh.faces.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{corpus, generate};

    fn base() -> (ShapeSpec, TriMesh) {
        let spec = corpus(true)[0].clone();
        let mesh = generate(&spec, 16, 32);
        (spec, mesh)
    }

    #[test]
    fn reorder_preserves_geometry_multiset() {
        let (_, mesh) = base();
        let mut rng = SmallRng::seed_from_u64(42);
        let shuffled = reorder(&mesh, &mut rng);
        assert_eq!(shuffled.vertices.len(), mesh.vertices.len());
        assert_eq!(shuffled.faces.len(), mesh.faces.len());
        // Same vertex multiset.
        let key = |v: &[f64; 3]| format!("{:.9}|{:.9}|{:.9}", v[0], v[1], v[2]);
        let mut a: Vec<String> = mesh.vertices.iter().map(key).collect();
        let mut b: Vec<String> = shuffled.vertices.iter().map(key).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
        // Same triangle multiset (by position triple, order-normalized).
        let tri_key = |m: &TriMesh, f: &[u32; 3]| {
            let mut ps: Vec<String> = f.iter().map(|&i| key(&m.vertices[i as usize])).collect();
            ps.sort();
            ps.join("&")
        };
        let mut ta: Vec<String> = mesh.faces.iter().map(|f| tri_key(&mesh, f)).collect();
        let mut tb: Vec<String> = shuffled
            .faces
            .iter()
            .map(|f| tri_key(&shuffled, f))
            .collect();
        ta.sort();
        tb.sort();
        assert_eq!(ta, tb);
    }

    #[test]
    fn rigid_preserves_pairwise_distances() {
        let (_, mesh) = base();
        let mut rng = SmallRng::seed_from_u64(7);
        let moved = rigid(&mesh, &mut rng);
        let d = |m: &TriMesh, i: usize, j: usize| -> f64 {
            let a = m.vertices[i];
            let b = m.vertices[j];
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        for &(i, j) in &[(0usize, 100usize), (5, 200), (50, 400)] {
            assert!((d(&mesh, i, j) - d(&moved, i, j)).abs() < 1e-9);
        }
    }

    #[test]
    fn variants_are_deterministic() {
        let (spec, mesh) = base();
        let a = variants(&spec, &mesh, 2);
        let b = variants(&spec, &mesh, 2);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.label, y.label);
            assert_eq!(x.mesh.vertices, y.mesh.vertices);
        }
    }

    #[test]
    fn variant_suite_covers_all_kinds() {
        let (spec, mesh) = base();
        let vs = variants(&spec, &mesh, 1);
        let kinds: std::collections::BTreeSet<&str> = vs.iter().map(|v| v.kind).collect();
        let expected: std::collections::BTreeSet<&str> = [
            "control", "rigid", "reorder", "remesh", "scale", "noise", "stretch", "tamper",
        ]
        .into_iter()
        .collect();
        assert_eq!(kinds, expected);
    }
}
