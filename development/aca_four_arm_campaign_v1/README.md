# Four-arm formal campaign

This separate orchestrator calls the unchanged `aca_four_arm_v1.train` entry point.
The twelve readiness checks pin that trainer and its protocol. The queue never
imports historical checkpoints, retries, resumes, changes hyperparameters, or opens test.
This is a project controlled ablation, not a verified exact author reproduction.

Explicit launch from this worktree with the verified environment:

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/tsai/miniconda3/envs/yolo11/bin/python -u -B \
  -m development.aca_four_arm_campaign_v1.queue \
  --source-root /home/tsai/projects/basketball --launch --accept-candidate-architecture
```

The default command starts nothing. A single inherited file lock prevents concurrent
campaigns. A matching existing campaign plan also blocks a duplicate launch, even
after interruption. Recovery needs separate inspection and explicit authorization;
there is no automatic resume command here.

Order: seed 20261024, 20261025, 20261026; within each seed baseline, LSTA, TSCI, ACA.
Each run starts fresh from pinned ImageNet V1; paired head and overlapping module
initialization, max 30 epochs, patience 5, strict minimum complete validation CE.
Each train epoch contains 7193 videos / 450 optimizer updates; validation contains
1547 videos. Individual early stopping advances to the next job. A failed or
interrupted run stops the campaign, including a zero exit without a complete summary.

`artifacts/campaign_*/plan.json` pins the common plan, all source/readiness files,
the environment, and this supervisor. `launcher.json`, `spawn.json`, and `state.json`
record PID, Linux process start ticks, boot ID, and full command line. Cross-check
all four when inspecting liveness; stale files after a reboot do not prove a live job.
Per-job logs contain exactly one `OUTPUT:` line identifying the fresh formal run.
State refreshes every 10 seconds, but progress is the last logged checkpoint
(every 25 optimizer updates), not an exact instantaneous count. A full train cursor
without a completed history row means validation is in progress. Check process
identity and GPU state before diagnosing stale logs. No validation score exists
before a complete epoch. Training loss and readiness metrics are not formal results.

Final per-run results must match the arm/seed/plan, full history, selection rule,
450 updates per completed epoch, and saved best/last checkpoints. After all twelve
complete, `summary.json` contains per-run selected validation metrics and per-arm
three-seed means/sample standard deviations, including Miss and Make recall.
It never automatically selects an overall variant or runs test. The project has
prior test exposure; these new results are validation only.

SIGTERM/SIGINT requests a safe trainer stop at an optimizer boundary, then leaves
the remaining queue unstarted. Closing the launching terminal does not send these
signals because the supervisor owns a detached session. Host shutdown still
interrupts training; checkpoints are atomic and no service restarts it automatically.

Verification:

```sh
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES='' \
  /home/tsai/miniconda3/envs/yolo11/bin/python -B -m unittest \
  development.aca_four_arm_campaign_v1.test_queue -v
```
