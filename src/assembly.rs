//! Multi-shell "assembly" corpus + articulation operators.
//!
//! Everything in `corpus.rs` is a single star-shaped solid; nothing there can
//! make a disconnected multi-part mesh or move one part relative to the rest.
//! This module fills that gap with a box-assembly "car" (body + four wheels)
//! and the perturbations that only make sense on it: steering a sub-part,
//! sliding it, resizing it.
//!
//! It exercises spectral3d's `check_consistent_shells` path (untouched by the
//! star corpus) and, more pointedly, probes how identity behaves when a *part*
//! of a multi-part asset is rearranged. Small moves should keep the identity
//! (quantization tolerance); large ones cross into a genuinely different solid.
//! The crossover is a curve, not a verdict, so every articulation variant is
//! `Expectation::Gray` — reported, never pass/fail. The crossover magnitude is
//! exactly the "how loud must a reconfiguration be before it mints a new
//! identity" threshold (the multi-shell Sybil deadband).

use crate::obj_io::TriMesh;
use crate::perturb::{Expectation, Variant};

/// One box shell. `yaw` spins it about its own vertical (Y) axis — the steering
/// motion. `steer` marks the parts the articulation operators are allowed to move.
#[derive(Debug, Clone)]
pub struct Part {
    pub center: [f64; 3],
    pub half: [f64; 3],
    pub yaw: f64,
    pub steer: bool,
}

/// A multi-shell assembly: a named bag of box parts, each its own closed shell.
#[derive(Debug, Clone)]
pub struct AssemblySpec {
    pub name: String,
    pub parts: Vec<Part>,
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rotate a point about the Y (vertical) axis.
fn yaw_y(p: [f64; 3], c: f64, s: f64) -> [f64; 3] {
    [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c]
}

/// Append one box shell — 8 corners, 12 outward-wound triangles — to `mesh`.
/// Winding is fixed per triangle by flipping any whose normal faces inward, so
/// the shell always reads as a positive-volume closed manifold regardless of
/// the quad listing. That keeps `check_consistent_shells` happy: every shell
/// winds the same way.
fn push_box(mesh: &mut TriMesh, part: &Part) {
    let (c, s) = (part.yaw.cos(), part.yaw.sin());
    let [cx, cy, cz] = part.center;
    let [hx, hy, hz] = part.half;
    let base = mesh.vertices.len() as u32;

    // Corner index packs the three sign bits: (sx_bit<<2)|(sy_bit<<1)|sz_bit.
    let mut corners = [[0.0f64; 3]; 8];
    for (bx, &sx) in [-1.0f64, 1.0].iter().enumerate() {
        for (by, &sy) in [-1.0f64, 1.0].iter().enumerate() {
            for (bz, &sz) in [-1.0f64, 1.0].iter().enumerate() {
                let local = yaw_y([sx * hx, sy * hy, sz * hz], c, s);
                corners[bx << 2 | by << 1 | bz] = [cx + local[0], cy + local[1], cz + local[2]];
            }
        }
    }
    let id = |a: usize, b: usize, d: usize| a << 2 | b << 1 | d;
    let quads = [
        [id(0, 0, 0), id(0, 1, 0), id(0, 1, 1), id(0, 0, 1)],
        [id(1, 0, 0), id(1, 1, 0), id(1, 1, 1), id(1, 0, 1)],
        [id(0, 0, 0), id(1, 0, 0), id(1, 0, 1), id(0, 0, 1)],
        [id(0, 1, 0), id(1, 1, 0), id(1, 1, 1), id(0, 1, 1)],
        [id(0, 0, 0), id(1, 0, 0), id(1, 1, 0), id(0, 1, 0)],
        [id(0, 0, 1), id(1, 0, 1), id(1, 1, 1), id(0, 1, 1)],
    ];
    for q in quads {
        for [a, b, d] in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let (pa, pb, pc) = (corners[a], corners[b], corners[d]);
            let n = cross(sub(pb, pa), sub(pc, pa));
            let tc = [
                (pa[0] + pb[0] + pc[0]) / 3.0,
                (pa[1] + pb[1] + pc[1]) / 3.0,
                (pa[2] + pb[2] + pc[2]) / 3.0,
            ];
            let (a, b, d) = if dot(n, sub(tc, part.center)) < 0.0 {
                (a, d, b)
            } else {
                (a, b, d)
            };
            mesh.faces
                .push([base + a as u32, base + b as u32, base + d as u32]);
        }
    }
    for corner in corners {
        mesh.vertices.push(corner);
    }
}

/// Concatenate every part's shell into one multi-shell mesh.
pub fn build(spec: &AssemblySpec) -> TriMesh {
    let mut mesh = TriMesh {
        vertices: Vec::new(),
        faces: Vec::new(),
    };
    for part in &spec.parts {
        push_box(&mut mesh, part);
    }
    mesh
}

/// A car: an elongated body (anisotropic enough to clear the WeakShape gate)
/// plus four wheels hung below it. The two front wheels are steerable; the
/// wheels sit clear of the body so the shells stay disjoint.
fn car(name: &str) -> AssemblySpec {
    let wheel = |x: f64, z: f64, steer: bool| Part {
        center: [x, -0.9, z],
        half: [0.5, 0.35, 0.18],
        yaw: 0.0,
        steer,
    };
    AssemblySpec {
        name: name.into(),
        parts: vec![
            Part {
                center: [0.0, 0.0, 0.0],
                half: [2.0, 0.5, 1.0],
                yaw: 0.0,
                steer: false,
            },
            wheel(1.5, 1.3, true),   // front-left  (steers)
            wheel(1.5, -1.3, true),  // front-right (steers)
            wheel(-1.5, 1.3, false), // rear-left
            wheel(-1.5, -1.3, false),
        ],
    }
}

/// Deterministic assembly corpus. `quick` keeps just the one car.
pub fn assemblies(quick: bool) -> Vec<AssemblySpec> {
    let mut out = vec![car("car")];
    if !quick {
        // Bigger wheels: a steered part that's a larger fraction of the whole,
        // so articulation should cross the deadband sooner.
        let mut big = car("car_bigwheel");
        for p in big.parts.iter_mut().filter(|p| p.steer) {
            p.half = [1.0, 0.7, 0.36];
        }
        out.push(big);
    }
    out
}

/// Clone the spec and apply `f` to every steerable part.
fn with(spec: &AssemblySpec, mut f: impl FnMut(&mut Part)) -> AssemblySpec {
    let mut s = spec.clone();
    for p in s.parts.iter_mut().filter(|p| p.steer) {
        f(p);
    }
    s
}

/// Articulation variant suite for one assembly. The base car registers; each
/// variant verifies back against it, so `id_kept` traces the deadband: it holds
/// (same identity) until the reconfiguration grows loud enough to cross into a
/// genuinely different solid, then drops.
pub fn variants(spec: &AssemblySpec, base: &TriMesh) -> Vec<Variant> {
    let mut out = vec![Variant {
        label: "control_identity".into(),
        kind: "control",
        magnitude: 0.0,
        expectation: Expectation::Same,
        mesh: base.clone(),
    }];

    // Steering: yaw the front wheels about their own vertical axis.
    for &deg in &[5.0f64, 15.0, 30.0, 45.0] {
        out.push(Variant {
            label: format!("steer_{deg}deg"),
            kind: "steer",
            magnitude: deg,
            expectation: Expectation::Gray,
            mesh: build(&with(spec, |p| p.yaw += deg.to_radians())),
        });
    }
    // Repositioning: slide the front wheels outward along +x.
    for &dx in &[0.05f64, 0.1, 0.2, 0.5] {
        out.push(Variant {
            label: format!("displace_{dx}"),
            kind: "displace",
            magnitude: dx,
            expectation: Expectation::Gray,
            mesh: build(&with(spec, |p| p.center[0] += dx)),
        });
    }
    // Resizing: scale the front wheels in place.
    for &k in &[1.5f64, 2.0, 3.0] {
        out.push(Variant {
            label: format!("partscale_{k}"),
            kind: "partscale",
            magnitude: k,
            expectation: Expectation::Gray,
            mesh: build(&with(spec, |p| p.half = [p.half[0] * k, p.half[1] * k, p.half[2] * k])),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assemblies_are_deterministic() {
        let a = assemblies(false);
        let b = assemblies(false);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.name, y.name);
            assert_eq!(x.parts.len(), y.parts.len());
        }
    }

    #[test]
    fn car_mesh_is_five_closed_shells() {
        let m = build(&car("t"));
        let v = m.vertices.len() as i64; // 5 boxes * 8
        let f = m.faces.len() as i64; // 5 boxes * 12
        let e = 3 * f / 2; // each edge shared by exactly 2 triangles
        assert_eq!(v, 40);
        assert_eq!(f, 60);
        // Euler sum over 5 disconnected genus-0 shells: sum(V-E+F) = 2 * 5.
        assert_eq!(v - e + f, 10);
    }

    /// The base car must clear every gate (closed shells, consistent winding,
    /// WeakShape). If it doesn't, the body's aspect ratio needs adjusting.
    #[test]
    fn spectral_accepts_base_car() {
        let m = build(&car("t"));
        let obj = crate::obj_io::to_obj_bytes(&m);
        let r = spectral3d::register(&obj, &spectral3d::SpectralParams::default());
        assert!(r.is_ok(), "base car should register, got {:?}", r.err());
    }

    #[test]
    fn variants_cover_articulation_kinds() {
        let car = car("t");
        let vs = variants(&car, &build(&car));
        let kinds: std::collections::BTreeSet<&str> = vs.iter().map(|v| v.kind).collect();
        assert!(kinds.contains("steer"));
        assert!(kinds.contains("displace"));
        assert!(kinds.contains("partscale"));
    }
}
