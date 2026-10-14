# Baseline v001 diagnosis and narrowly scoped v002 — 2026-10-12

## Evidence and scope

Read the saved SE01–SE04 `per_frame_v001` feature files and `action_rules_v001` predictions.
No pose inference, training, annotations, recording labels or splits were used as prediction
inputs. Diagnosis outputs are new under `outputs/action_baseline_diagnosis_20261012/`.
Old v001 code, tests, predictions and input artifacts remain unchanged, with checksum checks.

Artifacts in that directory:

- `diagnosis.json`: counts, ranges, exact moving/unknown intervals and coverage onset.
- `P001_SE0*_V001_diagnostic_rows.jsonl`: every frame's six statistics, coverage, branch,
  all thresholds and signed **value minus threshold** distances. Null statistics stay null.
- `P001_SE0*_V001_transitions.json`: previous/current values, margins and exact PTS for every
  transition, including all253 SE03 changes; not sampled or limited to examples.
- `rule_branch_intervals.json`: contiguous intervals for each reason, including motion-band
  uncertainty and posture conflicts. Each has first, last-included and next-exclusive PTS.
- `review_manifest.json`, full-resolution source PNGs and three `*_review_sheet.jpg` files:
  bounded qualitative review described below. No prediction overlays were inferred anew.
- `comparison.json`, `verification.json`: full v001/v002 counts, reasons, transitions and checks.
- `diagnose_v001.py`: audit/review generation script; fail rather than replace existing files.

Interval convention: frame indices **[start,end)**; corresponding PTS **[PTS_start,PTS_end)**
uses the next frame's presentation time, not a claim of an exact activity boundary. All listed
real inputs have time_base=1/30000. JSON includes exact integer PTS and last-included PTS too;
null end means end of input, not an invented last-frame duration. Seconds below are rounded
for reading; the integer/rational values are authoritative.

## Implementation audit before changing policy

Replayed v001 decisions exactly from all1,515 saved feature rows. Independently recomputed
all9,090 channel statistics from source adjacent intervals and verified metadata and hashes.
No discrepancy was found in duration weighting, clipping, masks or decision reproduction.
This is evidence against a window arithmetic defect on these inputs, not proof that every
possible malformed input is handled. No numeric threshold was adjusted.

Coverage = valid interval duration / configured1s, **not valid-frame fraction**. An interval
(ta,tb] contributes `tb-max(ta,t-1)` if positive. Static channels need both endpoint masks;
velocity needs the saved derivative mask, which already checks its pair. Values are weighted
by duration and divided by valid covered duration. The norm is taken **after** averaging the
velocity vector: net change can cancel when motion reverses. Box statistic is pixel W/H,
not normalized-image W/H. No future observations are used.

| Video | Typical adjacent dt (median) | First six-channel coverage pass | First known v001 label |
|---|---:|---|---|
| SE01 | .064s | f12,24001/30000=.800033s | same |
| SE02 | .064s | f12,24480/30000=.816000s | same |
| SE03 | .032s | f24,24480/30000=.816000s | f30,30243/30000=1.008100s |
| SE04 | .064s | f12,24002/30000=.800067s | same |

12 versus24 startup frames is therefore expected for roughly15 versus30 FPS. Synthetic
constant-feature15Hz,30Hz and irregular timelines all pass at .8s with equal covered duration
and motion statistic; minimum3intervals is also explicit and remains a separate constraint
at very low sampling rates. Right-endpoint geometry aggregation has sampling-dependent
approximation error for varying signals, but is the documented rule, not FPS substitution.
More observations also allow more threshold crossings to be counted; frame rate alone does
not establish why a particular source feature fluctuates.

How the six statistics affect decisions:

- Torso is upright at <=30deg; lying needs60..120deg and aspect>=1.2.
- Both knee means <=135deg support sitting; both>=155deg support standing. Intermediate
  knees do not force a posture. Missing any of the six channels blocks v001, even if that
  channel is not numerically used by the eventual posture branch.
- Hip and box vector-mean magnitudes are image fractions/s. Either>=.03 means high image
  motion; both<=.015 means quiet; the remaining band is indeterminate. These are not gait
  or real-world speed measurements.
- v001 lets motion-band uncertainty veto **every** posture, and high motion veto sitting/
  lying. Upright+high means moving even when only box size changes. These policy choices,
  rather than a coverage arithmetic error, account for the observed outputs.

## SE01: all38 moving predictions

All272 coverage-valid frames have standing geometry: torso .736..10.152deg, left knee
171.318..175.764deg, right knee167.671..174.874deg. Each moving frame uses
`upright_with_temporal_motion`;23 are hip-only triggers and15 box-only. No frame has both high.

| Frames | Exact PTS interval /30000 | Seconds [start,end) | Count | Trigger range | Distance above .03 |
|---|---|---|---:|---|---|
| [26,30) | [52322,60482) | [1.744067,2.016067) | 4 | hip .030007637.. .032374760 | +.000007637..+.002374760 |
| [115,130) | [232327,263047) | [7.744233,8.768233) | 15 | box .031343919.. .065064661 | +.001343919..+.035064661 |
| [160,176) | [323530,355691) | [10.784333,11.856367) | 16 | hip .030657049.. .056431503 | +.000657049..+.026431503 |
| [187,190) | [378251,384491) | [12.608367,12.816367) | 3 | hip .030185050.. .030877886 | +.000185050..+.000877886 |

For the box-triggered group the hip is only .000959725.. .005484448 (well below quiet .015).
At f120/PTS242407=8.080233s, six statistics are torso4.227deg, knees173.827/170.610deg,
aspect .411723, hip .001727442, box .065064661. Box is +.035064661 above high, while hip is
−.013272558 below quiet. Arm extension visible in the still triplet is compatible with
changing box width without walking. At f189 hip .030877886 is only +.000877886 above high.

Evidence: exact feature/branch values and still-image stance/arm differences. Hypothesis:
body sway, arm changes and/or pose/box jitter explain some triggers; sparse stills cannot
prove absence of any step throughout all38 frames. At the initial still-image diagnosis stage,
these were not confirmed errors; the subsequent user playback review below supplies separate evidence.

### SE01 v002: subsequent user playback review (2026-10-12)

The user reports viewing the video and confirms body adjustments **without walking steps** in
the half-open intervals [26,30),[115,130),[160,176),[187,190). Codex checked the saved predictions
and their indices/PTS; **Codex did not independently verify this visual observation**.
The intervals contain4+15+16+3=38frames, all predicted moving by v002, and exactly match its
saved moving indices. Under the project's definition of moving as walking with steps and
displacement, these reviewed predictions are recorded as **false_positive_moving** based on
user playback evidence. The exact PTS ranges are those in the SE01 table above.

Separate feedback/provenance and verification:
`outputs/P001_SE01_V001_baseline_v002_user_review_20261012/user_playback_feedback.json`
and `verification.json`. Prediction checksum matches original provenance; configuration,
predictions and existing annotations remain unchanged. No labels are inferred outside these
four intervals, no replacement activity labels are assigned, and no whole-video accuracy is
reported. This feedback does not itself change classifier rules or create another version.

## SE03: why253 changes occur

All564 frames with sufficient coverage have lying geometry. Torso93.833756..95.427849deg
is far inside [60,120]: at least33.833756deg above lower and24.572151deg below upper bound.
Aspect4.421703..4.524398 is at least+3.221703 above1.2. Knee means also remain valid.
Thus the posture branch never switches among the coverage-valid rows.

v001 output:334 quiet→lying,191 motion-band→unknown,39 high-motion/posture-conflict→unknown,
plus24 startup unknown. High triggers:33 hip-only,5 box-only,1 both. All253 label changes
are127 unknown→lying and126 lying→unknown. Their exact PTS and adjacent threshold distances
are in `P001_SE03_V001_transitions.json`; branch intervals are not treated as annotations.

| Example frame / PTS | Torso; knees L/R; aspect | Hip speed | Box speed | v001 branch and margin |
|---|---|---:|---:|---|
| f323 /327368 (10.912267s) | 94.281;175.687/174.915;4.481204 | .023911603 | .005392190 | band: hip +.008911603 above quiet, −.006088397 below high |
| f246 /249125 (8.304167s) | 94.463;175.583/175.683;4.468328 | .037824217 | .001917229 | conflict: hip +.007824217 above high |
| f298 /301927 (10.064233s) | 94.060;175.756/175.445;4.486962 | .004376955 | .006191511 | quiet: hip −.010623045 and box −.008808489 below quiet |

Evidence: unchanged posture support, branches driven by image-motion crossings, comparable
lying body configuration in the reviewed stills. Pose jitter is a plausible contributor, not
measured ground truth; small true body movement and pose error cannot be separated precisely
from this review. Regardless of cause, image motion by itself does not refute a lying posture.

## SE04: locomotion versus image change

All427 coverage-valid frames have standing geometry;393 high→moving,21 band→unknown,
13 quiet→standing. Moving is triggered by box alone215 times, both172 and hip alone6.
The rule has no foot-contact/gait evidence and cannot uniquely distinguish a turn, sway,
camera/pose jitter or changing silhouette from locomotion.

- f217/PTS438731=14.624367s: torso2.549deg, knees169.174/166.580deg, aspect .318819;
  hip .005545896 (−.009454104 below quiet), box .031563795 (+.001563795 above high).
  Triplet f202/217/224 shows orientation changing from back/side toward front with foot
  adjustment near the far end. A box-only moving label here does not establish walking.
- f224/PTS453132=15.104400s: torso2.575deg, knees172.836/168.966deg, aspect .276709;
  hip .013157110, box .029878231. Box is just −.000121769 below high but+.014878231 above
  quiet, producing unknown. Stills do not establish a stationary pause at this timestamp.
- f281/PTS568335=18.944500s: torso1.946deg, knees174.685/172.557deg, aspect .409727;
  hip .042855534 (+.012855534 above high), box .120985404 (+.090985404). Triplet f266/281/288
  shows the subject nearer/larger, compatible with approaching, not a confirmed interval.
- f18/PTS36480=1.216000s: hip .010611519, box .006682327 are below quiet; outputs standing.
  Initial stills cannot determine the exact first step; onset annotation remains pending.

These examples demonstrate ambiguity in the measured quantities. They do not quantify false
positives/negatives or settle prior pause/turn annotations. No upright locomotion threshold
change is justified by these sparse reviews alone, so v002 deliberately leaves SE01/SE04 unchanged.

## Bounded source review and provenance

Assistant inspected **three sheets,11 anchor examples,33 displayed slots/32 distinct source
frames**, each decoded sequentially with PyAV and verified against saved original index,
integer PTS and time_base. Full-resolution PNGs were saved; visual inspection used their
sheet renderings. This was **still-image review, not continuous1x playback**. No annotation
or frame-level ground truth was created. Source/PNG hashes and exact sample offsets are in
`review_manifest.json`; anchors have approximately−1s/current/+.5s contextual samples.

- SE01: one peak trigger from each of the four moving groups, anchors26/120/167/189.
  Subject remains upright in the displayed images; arm positions and weight distribution
  differ. No claim of continuous stationarity or absence of intervening steps.
- SE03: median-occurrence sample per band/conflict/quiet branch, anchors323/246/298.
  Subject appears lying in each sample; broadly similar outline. Joint accuracy unverified.
- SE04: quiet18, band224, box-only217 and both-high281, chosen by rule reason/channel.
  Images show initial stance, changing orientation and approaching as described above.

Selection was by failure/branch reason, not exhaustive review of253 transitions and not
selection of examples to maximize a score. No continuous playback was used to infer timing.

## One correction: v002 posture is not vetoed by image motion

`action_baseline_v2.BaselineV2` subclasses the unchanged v001 engine. When all original gates
pass and posture evidence is sitting/lying, v002 retains that posture even if image motion
is high or in the threshold band. This corrects conflation of a posture with absence of
image motion; it does not assert that the subject cannot be transported, adjust limbs or move
while sitting/lying. All other branches, numeric thresholds, coverage and window resets remain
unchanged. No smoothing, debounce, hold-last-label or extra temporal state was added.

To avoid hiding uncertainty, each row keeps the exact `v001_decision`, original `motion_evidence`,
and `image_motion_cautions` with band/conflict reasons when a posture overrides that veto.
For supported sitting/lying, `locomotion_evidence.status` is **unknown**, not a claim of
stationarity. Unsupported geometry or any missing channel/coverage/target remains unknown.
The correction applies equally to sitting and lying, irrespective of video identity.

This policy is justified by the distinction between posture and locomotion plus the observed
stable geometry/unstable motion veto. It is not a claim that all564 lying predictions are correct.
Sitting with high motion is tested synthetically; the current SE02 run does not test that case.

## Comparison and delay

Each cell shows **v001 → v002**; identical counts are written once. Counts use all frame rows.

| Video | Rows | standing | sitting | lying | moving | unknown | Transitions |
|---|---:|---:|---:|---:|---:|---:|---:|
| SE01 |284|195|0|0|38|51|19|
| SE02 |204|0|192|0|0|12|1|
| SE03 |588|0|0|334→564|0|254→24|253→1|
| SE04 |439|13|0|0|393|33|13|

Unknown reasons: SE01 startup12+band39 unchanged; SE02 startup12 unchanged; SE04 startup12+
band21 unchanged. SE03 startup24 remains unknown; band191/conflict39 are now lying **with those
230 cautions preserved**. All1,515 rows keep identical indices/timestamps, input values/masks,
six supporting statistics, windows and underlying posture/motion evidence. Only230 output labels
change. Fewer transitions is a direct consequence of policy, not evidence of better accuracy.

Added temporal delay: **0s**; same causal trailing1s window and80%coverage. First nonunknown
output SE01 .800033s,SE02 .816000s,SE04 .800067s is unchanged. SE03 moves from1.008100s (f30)
to .816000s (f24), .192100s earlier because the motion veto is removed. These are output
timestamps, not measured latency against a true action onset. Existing window aggregation
can still delay posture changes; no new delay or smoothing is hidden.

## Commands, checks and remaining work

New outputs: `outputs/P001_SE0{1,2,3,4}_V001_action_baseline_v002_20261012/`, each containing
`predictions.jsonl`, `summary.json`, `provenance.json`. v001 command and artifacts remain intact.
Example (already run; choose a new output directory to repeat):

```powershell
.\.venv\Scripts\python.exe -m patient_monitoring action-baseline-v2 outputs/P001_SE01_V001_features_v001_20261007/features.jsonl --output-dir outputs/P001_SE01_V001_action_baseline_v002_20261012
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_action_baseline*.py -v
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe -m pip check
```

Actual checks:23 focused tests PASS (15 existing+8 new), full129 tests PASS, pip check PASS;
four CLI runs complete. Tests cover sitting/lying with motion, preserved cautions, missing
channels, unresolved/gap/reseed resets, causal prefix,15/30Hz/irregular coverage, unchanged
upright rules and thresholds, no identity rules, output no-overwrite and partial failure.
483 pre-existing artifacts including old code/tests, outputs, data, weights, annotations,
PROJECT_PLAN and dependency declaration retain their hashes. STATUS remains at most60 lines.

Remaining: upright movement discrimination, geometric projection/pose jitter, uncertain human
boundaries, held-out subjects/sessions and accuracy evaluation. Complete independent timed
annotation before evaluating; review SE01 box-only group and SE04 turns with continuous
playback if refining locomotion later. Do not confirm annotations or choose thresholds from
these expected recording names. M2 stays IN_PROGRESS; no training, deployment or fall work.
