# Consumer surfaces

Two duck types sit on a `PulsarData` record: the **array surface** Enterprise
and Discovery already read, and the **linear engine** nltiming consumes as
the linear case of its timing-engine protocol.

This page is the readable API. [`SPEC.md`](../SPEC.md) is the normative
contract. psrdata does **not** define live timing-engine protocols; those live
in [nltiming](https://github.com/vhaasteren/nltiming)
(`nltiming.protocols`). This package implements the linear answers structurally
and does not import nltiming.

```python
from psrdata import PulsarData

psr = PulsarData.from_feather("J1909-3744.feather")
engine = psr.linear_engine()
```

---

## 1. The Enterprise / Discovery array surface

Likelihood readers treat a pulsar as named frozen arrays in the writer's row
order. Enterprise's `FeatherPulsar` and Discovery's `Pulsar` read the
Feather file with their own loaders and never import this package. The
in-memory record exposes the same names.

### Required by nltiming's `PulsarData` protocol

These are the fields a likelihood uses for the timing block:

| attribute | shape | meaning |
|---|---|---|
| `name` | `str` | pulsar name |
| `fitpars` | `tuple[str, ...]` | inferred parameters; also the `Mmat` column order |
| `toas` | `(n,)` | barycentric arrival times, TDB seconds |
| `residuals` | `(n,)` | timing residuals at the recorded reference, seconds |
| `toaerrs` | `(n,)` | unscaled TOA uncertainties, seconds |
| `freqs` | `(n,)` | barycentric frequency, MHz; `inf` when none was reported |
| `Mmat` | `(n, p)` | design matrix in fitter sign (`Δr ≈ −Mmat @ δ`) and PINT units |
| `flags` | `{str: (n,) str}` | TOA flags without the leading `-`; missing is `""` |
| `backend_flags` | `(n,)` | Enterprise backend label |

### Ephemeris extras

Optional for a minimal fake pulsar; present on every psrdata record, and used
by solar-system / geometry signal blocks (`nltiming.protocols.EphemerisExtras`):

| attribute | shape | meaning |
|---|---|---|
| `stoas` | `(n,)` | site arrival times, seconds |
| `telescope` | `(n,)` | observatory code |
| `pos` | `(3,)` | ICRS unit vector at the reference epoch |
| `pos_t` | `(n, 3)` | ICRS unit vector at each TOA |
| `sunssb` | `(n, 6)` | Sun position and, when available, velocity, lt-s |
| `planetssb` | `(n, 9, 6)` | planet positions and, when available, velocities, lt-s |
| `theta` | `float` | colatitude, `π/2 − dec`, radians |
| `phi` | `float` | right ascension, radians |
| `pdist` | `(float, float)` | distance and uncertainty, kpc |
| `dm` | `float` | model DM, pc cm⁻³; `0.0` when absent |
| `dmx` | mapping or `None` | Enterprise DMX table, or `None` when the model has no DMX |

### Record metadata (psrdata, not the Enterprise duck type)

These make the file self-describing. They are not required by the Enterprise
array protocol, but every `PulsarData` has them:

- `setpars` — every defined parameter, fitted or not
- `parameters` — `{name: ParameterFact(value, units, uncertainty)}`; values
  are decimal strings, units are PINT units
- `timing_package` — `{data_set_key: software that calculated residuals/Mmat}`
- `partim_compatibility` — `{data_set_key: "pint" | "tempo2"}`
- `residual_centering` — `{data_set_key: ResidualCentering}` (descriptive)
- `producer` — software that wrote the record (`"metapulsar"`, `"vela_jax"`, …)
- `extra` — optional additive producer metadata

A standalone record stores those three mappings under the reserved key
`"single"`. A combined record uses the producer’s data-set keys
(`"epta"`, `"ppta"`, …). Row `i` of every array is the writer’s row `i`;
psrdata never sorts.

---

## 2. The linear timing engine

```python
engine = psr.linear_engine()
# equivalent:
#   psrdata.linear_engine(psr)
#   LinearTimingEngine.from_pulsar_data(psr)
#   LinearTimingEngine.from_feather(path)
```

This is `Δr = −Mmat @ δ` over the record’s own matrix, in the shape
nltiming’s `TimingEngine` / `JacobianTimingEngine` protocols describe.

### Required operations

```python
engine.fitpars                      # tuple[str, ...]  == psr.fitpars
engine.native_units                 # {fitpar: PINT unit}, fitpar order
engine.residual_centering           # {data_set_key: ResidualCentering}

engine.reference_theta()            # float64 (p,) from the decimal strings
engine.reference_theta_exact()    # {fitpar: decimal string}  ← authority

engine.residual_delta(delta)        # −Mmat @ δ,  δ shape (p,)
engine.design_matrix(params=None)  # Mmat; any other params value is refused
engine.residual_jacobian()         # −Mmat
```

Units of `delta` are the PINT units in `native_units`. `float64`
`reference_theta()` is a derived view; digits that float64 cannot hold stay
in `reference_theta_exact()`.

### Linear-only answers

A live nonlinear engine fills these differently. The linear engine is
explicitly empty on the nonlinear surface:

```python
engine.engine_name                      # "linear"
engine.nonlinear_params                  # None
engine.identically_linear_fitpars()     # every fitpar
engine.binary_chart_capability(...)     # None
```

There is no `residual_delta_jax`. That method is nltiming’s
`JaxTimingEngine` protocol (NUTS / `jacfwd`). The linear engine stays on
NumPy; a consumer that wants JAX wraps `−M @ δ` in its own array library
(SPEC R-5.2.10).

### Combined records

A combined record does not store a partition. Each data set’s rows are the
support of its phase-offset column (`Offset` / `PHOFF`, or `Offset_<key>` /
`PHOFF_<key>`). Each contribution is the same engine duck type over that
block:

```python
engine.data_set_keys           # ("epta", "ppta") or ("single",)
engine.timing_package           # {"epta": "pint", ...}
engine.contributions()          # {key: LinearContribution}
# contribution.residual_delta(δ_local) matches engine.residual_delta(δ)[rows]
```

---

## 3. What this package is not

psrdata is not a `TimingPulsar`. That nltiming protocol is the record **plus**
a factory:

```python
pulsar.timing_engine(engines=...)   # returns a TimingEngine
pulsar.can_use_engines(engines=...)
pulsar.pint_model()                # may be None
```

A file-only consumer is: every array from the record, and
`timing_engine()` returning `record.linear_engine()`. Production
implementations (MetaPulsar, vela-jax) attach a live `r(θ)` instead.

Live protocols nltiming may also look for, and which the linear engine does
not implement:

| protocol | extra surface |
|---|---|
| `JaxTimingEngine` | `residual_delta_jax(δ)`, `precision_critical_fitpars()` |
| `TimingParameterMappingProvider` | `timing_parameter_mapping()` for composite name maps |
| `binary_chart_capability(...)` | returns a `BinaryChartCapability` or `None` |

See [`SPEC.md`](../SPEC.md) §5 for the linear-engine requirements and
[`SPEC-motivation.md`](../SPEC-motivation.md) for why the protocol stays in
nltiming.
