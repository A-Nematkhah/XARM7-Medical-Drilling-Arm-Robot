# Eval baselines

Tracked snapshots of perpendicular batch eval (sim). Working copies used by
`scripts/eval_batch.py` / `scripts/unlock_angled.py` stay gitignored under
`outputs/eval_batch.csv` and `config/angled_unlocked`.

## perp_seeds_0-9.csv

| Field | Value |
|-------|-------|
| Command | `python scripts/eval_batch.py --seeds 0-9 --min-success 0.8` |
| Success | 10/10 (100%) |
| Mean entry vs GT | ~2.2 mm |
| Mean depth (est) | ~25.8 mm |
| Date (UTC) | 2026-10-28 |

After a fresh local batch that meets `milestone.min_perp_*`, run:

```bat
python scripts/unlock_angled.py
```

`angled_unlocked.stamp.json` is a copy of the unlock stamp produced from this baseline.
