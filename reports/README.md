# Reports

Results of experiments, one report per experiment or per round, kept
separate from the plans (`PLAN-records.md`, `PLAN-problems.md`), which are
not edited by the experiments. A report states the build, the command, the
hardware, the budget, the numbers, and where the artifacts are. Results run
on the owner's machine (RTX 3060) are recorded here as they were shared.

| report | what |
|---|---|
| [2026-11-05-records-round.md](2026-11-05-records-round.md) | all results of the records round (sandbox and RTX 3060) |
| [2026-11-06-q7-tai256c-cpu-check.md](2026-11-06-q7-tai256c-cpu-check.md) | Q7 step 1: tai256c on the CPU host loop; distance detection |
| [2026-11-06-cvrp-data.md](2026-11-06-cvrp-data.md) | inventory of the CVRP data sets received for P7 |
| [2026-11-06-q6-q9.md](2026-11-06-q6-q9.md) | Q6 flags for `qap_gpu`; Q9 breakout local search, a cycling bug and its fix, the run |
| [2026-11-06-gpu-q6-sweep-and-a3.md](2026-11-06-gpu-q6-sweep-and-a3.md) | RTX 3060: Q6 aspiration sweep on tai50a (10n² reaches the best known on 5/5 seeds); DSJC500.5 at 47 with polish and VSS (1 conflict) |
| [2026-11-06-q6-tai50a-arms.md](2026-11-06-q6-tai50a-arms.md) | Q6 on tai50a: aspiration 20n² (5/5, median 35 s), arms A (4/5) and C (5/5) at 10n² |
| [2026-11-06-q6-aspiration-40-off.md](2026-11-06-q6-aspiration-40-off.md) | Q6 on tai50a: aspiration 40n² and off both 5/5 but median 503/466 s; 20n² (35 s) is kept |
| [2026-11-06-p7-acceptance-and-diagnostics.md](2026-11-06-p7-acceptance-and-diagnostics.md) | P7: CMT accepted (0.50%/0.26% vs 0.60/0.27%, 10 seeds); Golden 10⁶ ratio as the paper → GENI, implemented and tested |
| [2026-11-06-p7-geni-golden.md](2026-11-06-p7-geni-golden.md) | P7: GENI on Golden 1.82% (plain 2.10%, paper 1.30%), CMT unchanged; unstringing implemented and running |
| [2026-11-06-a2-preparation.md](2026-11-06-a2-preparation.md) | A2 preparation: CPU host loop on DSJC1000.5 (k = 82) and flat1000_76_0 (k = 81); the GPU will take the global-memory path |
| [2026-11-06-q9-bls-tai50a.md](2026-11-06-q9-bls-tai50a.md) | Q9: BLS on tai50a misses on 5/5 seeds (median 0.45%), retired; memetic baseline 2/4 so far |
| [2026-11-06-p7-steps-1-7.md](2026-11-06-p7-steps-1-7.md) | P7 steps 1 and 7: `.vrp` reader, `Routes` model, `verify_vrp.py`; all 212 received solutions recount in both |
| [2026-11-06-p7-its-first-results.md](2026-11-06-p7-its-first-results.md) | P7: sequential ITS, η = 10⁵, one seed: CMT mean gap 0.57% (paper 0.60%), Golden 2.65% (paper 1.30%) |
| [2026-11-06-a2-smoke-test.md](2026-11-06-a2-smoke-test.md) | A2 smoke test on the RTX 3060: DSJC1000.5 with global tables, clean, 1.2 M it/s; 8-bit table next |
| [2026-11-07-p7-golden-rebased.md](2026-11-07-p7-golden-rebased.md) | P7: Golden rebased onto one reference (the paper's best knowns moved 0.20%): the real deficit is 0.32 points at 10⁵, concentrated in Golden 3, 4, 18, 9 |
| [2026-11-07-p7-reference-fleet-phase-log.md](2026-11-07-p7-reference-fleet-phase-log.md) | P7: Golden 7/8 at the reference fleet: β pinned at its clamp in every phase, 0 feasible iterations; the fleet is tight (feasibility needs ≤ 3% gap) |
| [2026-11-07-p7-unstringing.md](2026-11-07-p7-unstringing.md) | P7: unstringing on Golden 1–4, 9–12: 1.87% against GENI alone 1.75% (paper 1.18%); Golden 3 and 4 not closed |
| [2026-11-07-p7-fleet-and-p.md](2026-11-07-p7-fleet-and-p.md) | P7: GENI without the spare vehicle on 11 Golden instances (2.48% against 1.76% with it; 1.41% against 1.65% without Golden 18, which ends at 13.14%); Golden 7/8 plain at 10⁶ with a tight fleet |
