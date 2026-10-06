# Learning Tracker — Intern Progress Log

Maintained by the mentor. Updated whenever graded checks happen.
Grading: ✅ solid · 🟨 partial · ❌ missed · ⬜ not yet assessed

## Concept scorecard

| Area | Status | Evidence |
|---|---|---|
| Newton/LM projection mechanics | 🟨 | predicted zero-step symptom correctly, missed Jacobian mechanism (2026-08-25) |
| Auto-vectorization mental model | ❌ | attributed to hardware limits; missed source-structure + runtime-codegen reasons |
| Threads vs SIMD | ❌ | conflated the two axes; corrected same day |
| Lexing fundamentals — lifetimes | ❌ | could not explain dangling `string_view` mechanism even after watching the ASan catch live (2026-08-26) |
| Lexing fundamentals — design coupling | 🟨 | sensed "incorrect tokens" outcome; missed the concrete breaking input and grammar-coupling principle |
| Lexing fundamentals — error recovery | ❌ | recalled "loops are standard" fact without invariant or failure mode |
| Motivation & scope articulation | ✅ | independently pushed back on AOT-vs-JIT framing — reviewer-grade objection |
| Modern-robotics landscape mapping | ✅ | drafted three-shifts analysis largely unaided |
| Multi-constraint linear algebra mechanics | ⬜ | not yet formally assessed |
| Kinematic tree topological lowering | ⬜ | not yet formally assessed |
| Control loop execution invariants | ⬜ | not yet formally assessed |

## Strengths observed

- Architectural skepticism: challenges design choices with correct instincts
  (the AOT-safety argument was genuinely defensible)
- Fast absorption of reframes once corrected — no repeated mistakes so far
- Asks the right meta-questions ("how does this map to X") before diving in

## Growth edges

- Mechanism vs symptom: tends to recall *what happens* without yet recalling
  *which formula causes it* — drill: derive, don't memorize
- Parallelism vocabulary: practice naming the axis (data/task/pipeline) when
  describing any speedup
- Dodges direct technical questions when unsure — in this internship, wrong
  attempts are graded kindly and count more than silence

## Graded checks — answered

**2026-08-26 · Lexer quiz (3 questions, issued 2026-08-25)**

| Question | Answer given | Grade |
|---|---|---|
| When do token `string_view`s dangle if a function returns tokens but not source? | "for the parser cause we need to implement the endpoint of function in it" | ❌ mechanism not engaged |
| What breaks if `-` always starts a number? | "would be considered as string not a value… incorrect tokens" | 🟨 outcome gesture, no concrete input/stage |
| Why loop instead of recursion in error path? | "recursion for each character is bad practice, looping is industrial standard" | ❌ rule recalled, invariant + failure mode missing |

Round result: **0 solid / 1 partial / 2 misses.**

---

### Re-teach delivered (same day)

1. Views are pointer+length into someone else's buffer; owner must outlive
   every borrower (CLI keeps `src`; test harness failed this and ASan proved it).
2. Lexical decisions encode grammar decisions: `-` joins numbers only when
   provably followed by a digit; revisiting is mandatory if arithmetic is ever
   added (`x - 3` would silently mis-tokenize as `x`, `-3`).
3. Invariant: `next()` returns after bounded work, consuming ≥1 char on error;
   recursion burned one stack frame per garbage byte → 10 MB of `@` = crash.
   The reason is bounded resource usage on hostile input, not convention.

## Mentor notes

- Learning goal stated explicitly: understand compiler design deeply, not
  compete with published systems — plan and scope reflect this consistently.
- 2026-08-26: pattern confirmed across two sessions now — facts retained,
  mechanisms lost. Pedagogy adjusted going forward: (a) memory-diagram style
  explanations before terminology, (b) one short question per concept during
  lessons rather than end-of-session batches, (c) require answers to name the
  *formula/mechanism*, not the outcome. Language-phrasing noise accounted for;
  grading targets content, not grammar.

