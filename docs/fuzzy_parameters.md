# Fuzzy Parameters

This repository uses the supplementary material parameters for the default
FuDU fuzzy inference system.

CLI presets:

```bash
fudu score --fuzzy-preset nuclear_fuel_rod ...
fudu score --fuzzy-preset eles ...
```

## Nuclear Fuel Rod Dataset

### Global Uncertainty Ug

| Variable | a | b | c | d | Range |
| --- | ---: | ---: | ---: | ---: | --- |
| VL | 0.0 | 0.0 | 0.1 | 0.2 | [0, 0.2) |
| L | 0.1 | 0.2 | 0.3 | 0.4 | [0.1, 0.4) |
| H | 0.3 | 0.4 | 0.6 | 0.8 | [0.3, 0.8) |
| VH | 0.6 | 0.8 | 1.0 | 1.0 | [0.6, 1.0] |

### Defect Uncertainty Ud

| Variable | a | b | c | d | Range |
| --- | ---: | ---: | ---: | ---: | --- |
| VL | 0.0 | 0.0 | 0.1 | 0.2 | [0, 0.2) |
| L | 0.1 | 0.2 | 0.5 | 0.6 | [0.1, 0.6) |
| H | 0.5 | 0.6 | 0.8 | 0.9 | [0.5, 0.9) |
| VH | 0.8 | 0.9 | 1.0 | 1.0 | [0.8, 1.0] |

## ELES Dataset

### Global Uncertainty Ug

| Variable | a | b | c | d | Range |
| --- | ---: | ---: | ---: | ---: | --- |
| VL | 0.0 | 0.0 | 0.1 | 0.2 | [0, 0.2) |
| L | 0.1 | 0.2 | 0.3 | 0.4 | [0.1, 0.4) |
| H | 0.3 | 0.4 | 0.55 | 0.65 | [0.3, 0.65) |
| VH | 0.55 | 0.65 | 1.0 | 1.0 | [0.55, 1.0] |

### Defect Uncertainty Ud

| Variable | a | b | c | d | Range |
| --- | ---: | ---: | ---: | ---: | --- |
| VL | 0.0 | 0.0 | 0.08 | 0.15 | [0, 0.15) |
| L | 0.08 | 0.15 | 0.35 | 0.45 | [0.08, 0.45) |
| H | 0.35 | 0.45 | 0.70 | 0.80 | [0.35, 0.80) |
| VH | 0.70 | 0.80 | 1.0 | 1.0 | [0.70, 1.0] |

## Rule Base

Rows are `Ug`, columns are `Ud`.

| Ug / Ud | VL | L | H | VH |
| --- | --- | --- | --- | --- |
| VL | DNS | LS | HS | MS |
| L | LS | LS | HS | MS |
| H | HS | HS | MS | MS |
| VH | MS | MS | MS | MS |

## Sampling Probability

| Action | Probability |
| --- | ---: |
| DNS | 0.0 |
| LS | 0.1 |
| HS | 0.5 |
| MS | 1.0 |

When multiple output actions have the same membership strength, FuDU uses the
safety-first priority `MS > HS > LS > DNS`.
