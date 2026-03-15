# Naturalness Roadmap

This document captures the long-term plan for making Ummfiltered outputs feel genuinely natural, not just technically cleaned up.

The core product goal is:

> Make it sound like the speaker naturally said it that way the first time.

## Principles

- Optimize for sentence flow, not just local seam quality.
- Preserve cadence, breaths, emphasis, and emotional shape.
- Prefer original source audio over synthetic repair whenever possible.
- Use AI surgically, not as the default answer to every seam.
- Never treat `0` fillers as success if the edited line loses intended words or sounds obviously cut.

## V2: Natural Editing Foundation

Goal: Make edits feel good enough that most people stop noticing them.

North star: A listener should usually not be able to tell where the edit happened.

### Priority work

- [ ] Move from filler-by-filler cutting to phrase-aware editing.
- [ ] Preserve cadence and breaths around edits.
- [ ] Add `delete vs shorten vs mask` decisioning instead of always fully removing fillers.
- [ ] Expand seam candidate search with more timing and pause options.
- [ ] Improve source-audio patching for tiny missing words near cuts.
- [ ] Add stronger “don’t over-compress this sentence” guardrails.
- [ ] Improve seam reports and QA tooling for listening review.
- [ ] Add a naturalness critic based on hand-designed features plus listening tests.

### Execution Program

#### 1. Phrase-aware editing

Desired user-facing outcome:
Lines should still sound like a coherent spoken thought instead of a sequence of cleanup cuts.

Engineering behaviors to build:
- Group nearby filler decisions into phrase windows instead of planning each cut in isolation.
- Let each filler choose between `delete`, `shorten`, and `mask` instead of always fully removing it.
- Score edits at phrase level so local seam wins do not create awkward phrase rhythm.
- Reject phrase plans that preserve technical cleanliness but make the sentence feel rushed or unnaturally tight.

How success will be judged:
- Fewer phrase-level timing complaints in listening review.
- Better transcript preservation around dense filler clusters.
- Lower worst-case naturalness scores on multi-filler phrases.

#### 2. Cadence and breath preservation

Desired user-facing outcome:
After editing, the speaker should still sound like they are breathing and pacing thoughts naturally.

Engineering behaviors to build:
- Detect breaths, inhale noise, release tails, and micro-pauses near edits.
- Preserve nearby breath structure whenever possible instead of flattening it away.
- Only add timing back when compression makes the phrase sound rushed.
- Add cadence guards that reject over-compressed sentences even if seam scores are otherwise good.

How success will be judged:
- Fewer “coppy” or rushed phrases on listening review.
- Improved phrase-level timing metrics compared with current baseline.
- Fewer accepted edits where the sentence loses natural recovery after a removed filler.

#### 3. Better seam placement

Desired user-facing outcome:
If a cut must happen, it should land where a human editor would try to hide it.

Engineering behaviors to build:
- Search seam anchors for low-energy, zero-crossing, breath-boundary, and syllable-safe positions.
- Generate multiple seam candidates per boundary rather than committing to one narrow strategy.
- Use different seam logic for speech-to-speech, speech-to-breath, breath-to-speech, and low-energy transitions.
- Reject seam candidates that are locally smooth but worsen phrase flow or increase transcript risk.

How success will be judged:
- Lower seam median and p95 across the golden set.
- Fewer obvious pops, fades, and chopped word starts in manual review.
- Better worst-seam performance on `test1`, `test2`, and `test3`.

#### 4. Source-audio healing

Desired user-facing outcome:
Tiny lost or clipped words should be restored using the original speaker whenever possible.

Engineering behaviors to build:
- Improve source-context patching for clipped starts, clipped ends, and missing short words near cuts.
- Align source patches to local timing and energy so they feel like part of the same take.
- Prefer source-audio repair before any synthetic replacement strategy.
- Treat this as the primary V2 repair path for transcript-preserving naturalness.

How success will be judged:
- Fewer missing or damaged words near cut boundaries.
- Better preservation of exact non-filler transcript on real files.
- Clear reduction in “word got swallowed by the cut” failures during rerender review.

#### 5. Naturalness critic and review tooling

Desired user-facing outcome:
The system should consistently choose the version that sounds most human, and bad outputs should be easy to inspect.

Engineering behaviors to build:
- Expand scoring to include seam audibility, cadence compression, breath continuity, pitch/energy continuity, and transcript-risk signals.
- Rank phrase candidates and seam candidates with the critic, not just local amplitude heuristics.
- Save worst seams and worst phrases per run with timestamps, scores, and transcript diffs.
- Make before/after candidate comparison easy enough to support repeated listening review.

How success will be judged:
- Critic rankings align better with what manual review flags as unnatural.
- Faster debugging of the worst failures in the golden set.
- Lower recurrence of the same dominant unnaturalness mode over repeated iterations.

### Expected outcome

- Fewer obvious chops and fades.
- Better transcript preservation.
- More human-sounding rhythm after filler removal.
- Strong local-only editing quality.

### V2 Testing And Acceptance

#### Automated regression tests

- Keep unit tests for cut safety, transcript preservation, source-audio healing, breath preservation, and cadence guard behavior.
- Add integration tests for dense filler speech, fillers near breaths, fast speech with little pause room, and phrase-level delete vs shorten vs mask behavior.
- Keep regression coverage for “0 fillers but missing transcript,” clipped words near cuts, over-smoothed fades, and audio disappearing after early edits.

#### Golden real-video dataset

- Use `test1`, `test2`, and `test3` as the initial acceptance set.
- Expand the dataset with fast speech, dense filler clips, breath-heavy clips, noisy-room clips, and stereo recordings.
- Keep a reference transcript, filler annotations, and listening notes for each golden clip.

#### Hard acceptance gates

- Fillers are removed or intentionally reduced by policy rather than accidentally left behind.
- The intended non-filler transcript is preserved exactly.
- No change is accepted if median or p95 naturalness regresses across the golden set.
- Channel count and basic output integrity must remain correct for every accepted render.

#### Human listening review

- Listen to the worst 5 seams and worst 5 phrases for every golden-set run.
- Compare against the previous best output, ideally with blind A/B review when practical.
- Reject any output with obvious pops, fades, robotic patches, rushed cadence, or missing-breath artifacts.

#### Iteration loop

1. Run the full golden set.
2. Rank the worst failures by severity and recurrence.
3. Fix the single dominant unnaturalness mode.
4. Re-run the full golden set.
5. Keep the change only if overall naturalness improves without transcript regression.
6. Repeat until there is no clearly dominant recurring unnaturalness failure mode.

### V2 Done Means

V2 is not done until all of these are true:

- Fillers are removed or intentionally reduced in a way that still sounds natural.
- The non-filler transcript is preserved exactly.
- The worst seams no longer sound obviously edited on manual review.
- `test1`, `test2`, and `test3` all pass both objective checks and listening-based review.
- There is no still-dominant recurring unnaturalness failure mode across the golden set.

`0` fillers alone is not success.

## V3: AI-Assisted Speech Repair

Goal: Move from seam cleanup to true speech reconstruction.

V2 must first establish a strong deterministic and source-audio baseline before heavier AI repair becomes the primary quality lever.

### Priority work

- [ ] Add context-conditioned speech inpainting as the primary repair backend.
- [ ] Repair exact-text short phrases using left and right real audio context.
- [ ] Match prosody: pitch contour, energy contour, timing, and breath placement.
- [ ] Generate multiple repair candidates per seam or phrase.
- [ ] Add speaker adaptation from the source clip.
- [ ] Plan edits at the sentence level instead of isolated seam level.
- [ ] Use a critic model to rank “natural vs edited” candidates.

### Expected outcome

- Many repaired edits sound like continuous original speech.
- AI becomes a differentiated quality layer, not just a patch.
- Ummfiltered becomes meaningfully stronger than simple cut-based editors.

## Moonshot: Clean-Take Generation

Goal: Turn Ummfiltered into a performance reconstruction system.

### Priority work

- [ ] Clean up more than fillers: false starts, repetitions, corrections, verbal clutter.
- [ ] Build an “intended performance” representation for each sentence or phrase.
- [ ] Regenerate a clean spoken version in the speaker’s style.
- [ ] Add joint audio/video repair for lip consistency and motion continuity.
- [ ] Build an end-to-end naturalness critic over both audio and video.
- [ ] Add a review workflow with multiple candidate deliveries per line.

### Expected outcome

- Output feels like a clean original take.
- Product shifts from filler remover to post-production performance editor.

## Recommended Build Order

1. V2 first:
   Phrase planning, cadence preservation, source-audio repair, stronger guardrails.
2. V3 second:
   Context-conditioned speech repair, speaker adaptation, critic-guided selection.
3. Moonshot last:
   Full clean-take generation and joint audio/video reconstruction.

## What To Build First

If we want the highest-impact next steps, build these first:

1. Phrase-level edit planning
2. Breath and cadence preservation
3. Source-audio healing for lost words
4. Naturalness critic for candidate ranking
5. Context-conditioned speech inpainting

## Tracking Notes

- This roadmap is intentionally outcome-first, not file-first.
- When work begins on any item above, add implementation notes and links to issues or PRs under the relevant section.
- If a new idea does not help speech feel more like an original uninterrupted take, it is probably not core to this roadmap.
