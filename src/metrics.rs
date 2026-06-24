//! Result rows, CSV serialization, and aggregation into the two headline
//! numbers: stability (same-expected variants keeping their identity) and
//! discrimination (different-expected variants and distinct base shapes
//! NOT sharing an identity).
//!
//! Identity predicate: registration stores `(id, helper)`. Verification of a
//! variant uses the base helper and succeeds when the recovered id equals the
//! registered id. So the single source of truth is `id_kept`.

use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct Row {
    pub shape: String,
    /// base | rigid | reorder | remesh | scale | noise | stretch | tamper
    pub kind: String,
    pub label: String,
    pub magnitude: f64,
    /// same | different | gray | - (base rows)
    pub expectation: String,
    /// ok | rejected | error
    pub outcome: String,
    pub elapsed_ms: u128,
    /// Variant verified back to the base registration id.
    pub id_kept: Option<bool>,
}

pub const CSV_HEADER: &str = "shape,kind,label,magnitude,expectation,outcome,elapsed_ms,id_kept";

impl Row {
    pub fn to_csv(&self) -> String {
        fn opt_bool(v: Option<bool>) -> String {
            v.map(|b| (b as u8).to_string()).unwrap_or_default()
        }
        format!(
            "{},{},{},{},{},{},{},{}",
            self.shape,
            self.kind,
            self.label,
            self.magnitude,
            self.expectation,
            self.outcome,
            self.elapsed_ms,
            opt_bool(self.id_kept),
        )
    }
}

/// Compare a recovered variant id against the registered base id.
pub fn id_kept(registered: &str, recovered: &str) -> bool {
    registered == recovered
}

/// Aggregated stability/discrimination table grouped by (kind, magnitude).
pub fn summarize(rows: &[Row]) -> String {
    #[derive(Default)]
    struct Acc {
        n: usize,
        ok: usize,
        error_or_rejected: usize,
        kept: usize,
        expectation: String,
    }

    let mut groups: BTreeMap<(String, String), Acc> = BTreeMap::new();
    for r in rows.iter().filter(|r| r.kind != "base") {
        let key = (r.kind.clone(), format!("{}", r.magnitude));
        let acc = groups.entry(key).or_default();
        acc.n += 1;
        acc.expectation = r.expectation.clone();
        match r.outcome.as_str() {
            "ok" => acc.ok += 1,
            _ => acc.error_or_rejected += 1,
        }
        acc.kept += (r.id_kept == Some(true)) as usize;
    }

    let mut s = String::new();
    s.push_str(&format!(
        "{:<9} {:>7} {:<9} {:>4} {:>4} {:>5} {:>7}\n",
        "kind", "mag", "expect", "n", "ok", "err", "id-kept"
    ));
    for ((kind, mag), a) in &groups {
        let denom = a.ok.max(1);
        s.push_str(&format!(
            "{:<9} {:>7} {:<9} {:>4} {:>4} {:>5} {:>6}%\n",
            kind,
            mag,
            a.expectation,
            a.n,
            a.ok,
            a.error_or_rejected,
            100 * a.kept / denom,
        ));
    }
    s
}

/// Pairwise identity collisions between distinct registered base shapes.
pub fn collision_report(bases: &[(String, String)]) -> String {
    let mut s = String::new();
    let mut collisions = 0;
    for i in 0..bases.len() {
        for j in i + 1..bases.len() {
            let (na, ida) = &bases[i];
            let (nb, idb) = &bases[j];
            if ida == idb {
                collisions += 1;
                s.push_str(&format!("  COLLISION {na} <-> {nb}: same id {ida}\n"));
            }
        }
    }
    if collisions == 0 {
        s.push_str("  none\n");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_kept_compares_registered_and_recovered_ids() {
        assert!(id_kept("registered", "registered"));
        assert!(!id_kept("registered", "different"));
    }

    #[test]
    fn collision_detects_shared_id() {
        let bases = vec![
            ("s1".to_string(), "a".to_string()),
            ("s2".to_string(), "a".to_string()),
            ("s3".to_string(), "d".to_string()),
        ];
        let report = collision_report(&bases);
        assert!(report.contains("s1 <-> s2"));
        assert!(!report.contains("s3"));
    }
}
