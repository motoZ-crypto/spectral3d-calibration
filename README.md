# Spectral3d Calibration

A measurement bench for the `spectral3d` register/verify pipeline. It takes a
deterministic corpus of procedural shapes, perturbs each one the way a real
scanner would, and reports how often the perturbed mesh still verifies back to
the registered identity.

## What this is, and what it isn't

This is an **instrument**, not a test suite. It produces numbers and curves you
read by hand. You run it with `cargo run --release` when you tune the quantizer,
swap a feature, or want fresh stability and discrimination figures.

The **pass/fail contract** lives elsewhere. `spectral3d`'s own `cargo test`
(the `e2e` module) asserts the headline promises: rigid poses keep identity,
stretch changes it, weak shapes get rejected, the golden hash holds. Those
assertions are the tripwire CI watches. This bench does not duplicate them.

`cargo test` here runs only the bench's **self-checks** (see below).

## What it measures

Every shape is registered once to get `(id, helper)`. Each perturbation is then
verified against that helper. The headline metric is **id-kept**: the recovered
id equals the registered id.

| Kind      | Simulates                                | Expectation     |
| --------- | ---------------------------------------- | --------------- |
| `control` | the untouched mesh, re-serialized        | same (100%)     |
| `rigid`   | the object in a different pose           | same (100%)     |
| `reorder` | same geometry, shuffled vertex/face order| same (100%)     |
| `remesh`  | the same surface at a different sampling  | same (100%)     |
| `scale`   | a unit-convention change (mm vs cm)      | same (100%)     |
| `noise`   | radial scan noise on the surface         | same (100%)     |
| `stretch` | a genuinely different object (z-stretch) | different (0%)  |
| `tamper`  | a local bump, the forgery probe          | gray, no verdict|
| `steer`   | an assembly sub-part rotated in place    | gray, no verdict|
| `displace`| an assembly sub-part slid to a new spot  | gray, no verdict|
| `partscale`| an assembly sub-part resized            | gray, no verdict|

`control` doubles as the in-run determinism canary. Anything under 100% there
means the bench itself drifted, not the algorithm.

Alongside the per-kind table, the run prints a **cross-shape collision report**:
distinct base shapes must land on distinct ids.

## The corpus

Four groups, each a star-shaped radial solid so any resolution regenerates the
same geometry:

- **rocks** — asymmetric bumpy superellipsoids, the algorithm's intended diet.
- **bumpy** — bumpy spheres, weaker anisotropy, harder for PCA.
- **boxes** — an aspect-ratio family (cube, plate, brick), the regular-shape and
  PCA-degenerate traps.
- **nearsym** — near-symmetric shapes that trip the PCA sign and order cliff.

`--quick` keeps one shape per group.

The `spectral3d` shape gate rejects weak, near-regular, or degenerate solids at
registration. That is a valid outcome, reported as `rejected`, and separate from
any verification drift.

## How to use

```sh
# Full stability / discrimination run, 2 seeds per stochastic perturbation
cargo run --release -- --seeds 2 --out results/default_seeds2.csv

# Quick smoke run: one shape per group, single seed
cargo run --release -- --quick --out results/quick.csv

# Feature-drift diagnostic (no hashing, see below)
cargo run --release -- --drift --seeds 2

# Sweep the quantizer bucket width
cargo run --release -- --quant-scale 1.5 --out results/scale_1p5.csv

# Bench self-checks only
cargo test --release
```

Debug-mode float math is slow. Prefer `--release` everywhere.

### Options

| Flag                | Meaning                                  | Default       |
| ------------------- | ---------------------------------------- | ------------- |
| `--quant-scale <X>` | multiplier on every spectral bucket width| `1.0`         |
| `--seeds <N>`       | random instances per stochastic kind     | `2`           |
| `--quick`           | one shape per group, single seed         | off           |
| `--drift`           | feature-drift diagnostic, skips hashing  | off           |
| `--out <PATH>`      | CSV destination                          | `results/results.csv` |

Generated CSVs are throwaway artifacts. `results/` ignores them by default. Keep
only a hand-reviewed summary in git, or pin one CSV as a release snapshot.

## Reading the output

The summary groups every variant by `(kind, magnitude)` and prints `n`, `ok`,
`err`, and `id-kept%`. Read it against the expectation column: `same` rows want
100%, `different` rows want 0%, `gray` rows are a curve you watch, not a gate.

### CSV columns

The CSV behind that summary has one row per base and per variant:

| Column        | Meaning                                                         |
| ------------- | --------------------------------------------------------------- |
| `shape`       | a star shape (`rock0`…) or an assembly (`car`, `car_bigwheel`)  |
| `kind`        | the perturbation family (`base`, `control`, `steer`, …)         |
| `label`       | the specific variant, e.g. `steer_15deg`, `displace_0.1`        |
| `magnitude`   | the kind's numeric knob (degrees, offset, or scale factor)      |
| `expectation` | `same`, `different`, `gray`, or `-` on base rows                |
| `outcome`     | `ok`, `rejected` (shape gate), or `error`                       |
| `elapsed_ms`  | register or verify time for that row, in milliseconds           |
| `id_kept`     | `1` recovered the base id, `0` did not, empty on base rows      |

`id_kept` is the column that matters. `1` means the perturbed mesh still verifies
as the same identity. `0` means it landed on a new one. `elapsed_ms` sits right
before it, so the last two numbers on a row are timing then identity. Do not
mistake the timing for the verdict.

## Drift mode

`--drift` skips hashing and reports, per feature dimension, the worst-case
movement across the corpus measured in bucket units (`drift / (step * scale)`).
This turns bucket-width tuning from guesswork into measurement:

- `same` rows want every stability-carrying dimension under roughly half a
  bucket once the registered helper offset is applied.
- `different` rows want at least one discriminating dimension past half a bucket.

The flip report marks each dimension that crossed a bucket boundary:

- `~` — the plain bucket changed, but the registered helper still recovers it.
- `!` — verification would land on a different bucket and lose the identity.

## Limits

- A perfect surface replica hashes identically. Pair shape identity with weight,
  material, or provenance when you need a stronger anti-copy guarantee.
- Noise versus tampering is a parameterized gray zone, not a clean boundary.
- The identity hash is deterministic and public. Treat it as an identity tag,
  not a secret or an auth token.
