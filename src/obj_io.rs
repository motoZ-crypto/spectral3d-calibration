//! Triangle mesh container and OBJ serialization (`v`/`f` records only —
//! the spectral parser ignores everything else).

/// Indexed triangle mesh, f64 positions, deterministic ordering.
#[derive(Debug, Clone)]
pub struct TriMesh {
    pub vertices: Vec<[f64; 3]>,
    pub faces: Vec<[u32; 3]>,
}

impl TriMesh {
    pub fn centroid(&self) -> [f64; 3] {
        let n = self.vertices.len().max(1) as f64;
        let mut c = [0.0; 3];
        for v in &self.vertices {
            c[0] += v[0];
            c[1] += v[1];
            c[2] += v[2];
        }
        [c[0] / n, c[1] / n, c[2] / n]
    }

    /// RMS distance of vertices from the vertex centroid (scale proxy).
    pub fn rms_radius(&self) -> f64 {
        let c = self.centroid();
        let n = self.vertices.len().max(1) as f64;
        let sum: f64 = self
            .vertices
            .iter()
            .map(|v| {
                let dx = v[0] - c[0];
                let dy = v[1] - c[1];
                let dz = v[2] - c[2];
                dx * dx + dy * dy + dz * dz
            })
            .sum();
        (sum / n).sqrt()
    }
}

/// Serialize to OBJ bytes with `v` and `f` records.
///
/// Keep the 7-decimal precision stable: it is part of what archived CSV
/// results were measured against.
pub fn to_obj_bytes(mesh: &TriMesh) -> Vec<u8> {
    use std::fmt::Write;
    let mut s = String::with_capacity(mesh.vertices.len() * 40 + mesh.faces.len() * 16);
    for v in &mesh.vertices {
        writeln!(s, "v {:.7} {:.7} {:.7}", v[0], v[1], v[2]).unwrap();
    }
    for f in &mesh.faces {
        // OBJ is 1-indexed.
        writeln!(s, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1).unwrap();
    }
    s.into_bytes()
}
