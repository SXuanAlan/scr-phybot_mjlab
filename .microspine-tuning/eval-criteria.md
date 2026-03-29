# Evaluation Criteria

## Hard Constraints (must ALL pass)
- **H1**: Phase 1 unchanged — 20N +x for 3s, no modifications
- **H2**: No anti-cheat (AC1-AC3 all clean)
- **H3**: Final z displacement < 0.5m (hand doesn't fall off wall)
- **H4**: The hand stays stable, do not vibrate.


## Soft Targets (count how many pass)
- **S1**: Final z displacement < 0.1m (minimal drift)
- **S2**: Max z drift during phase 2 < 0.15m
- **S3**: Phase 2 survival >= 7.0s (full episode)
- **S4**: Z stability std < 0.02m (steady state in last 2s)
- **S5**: Max x drift < 0.01m (stays pressed to wall) at phase 2
- **S6**: >= 100% of envs survive full phase 2
- **S7**: Actual RT of simulation should at least >=70% of Traget RT (The method should not slow down the simulation)

## Pass Criteria
- Verdict = PASS: H1-H4 all true AND >= 5/6 soft targets met
- Verdict = FAIL: H1-H4 all true but < 5/6 soft targets
- Verdict = PIVOT: H1-H4 violated OR 3 consecutive iterations with no improvement
