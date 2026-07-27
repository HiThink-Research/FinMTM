
<p align="center">
  <h1 align="center">
    <img src="static/logo.png"  height="40" style="position:relative; top:6px;">
    FinMTM: A Multi-Turn Multimodal Benchmark for Financial Reasoning and Agent Evaluation
  </h1>
</p>
  <p align="center">
       <strong>Chenxi Zhang</strong><sup>1,2*</sup>
    ,
    <strong>Ziliang Gan</strong><sup>1,3*</sup>
    ,
    <strong>Liyun Zhu</strong><sup>1*</sup>
    ,
    <strong>Youwei Pang</strong><sup>4</sup>
    ,
    <strong>Qing Zhang</strong><sup>5</sup>
    ,
     <strong>Rongjunchen Zhang</strong><sup>1 ♠</sup>
  </p>

<p align="center">
  <sup>1</sup> HiThink Research &emsp; <sup>2</sup>Wuhan University &emsp; <sup>3</sup>Zhejiang University &emsp; <sup>4</sup> Nanyang Technological University &emsp; <sup>5</sup>Shanghai Institute of Technology &emsp; <br>
  <sub><sup>*</sup>Equal Contribution &emsp; <sup>♠</sup>Corresponding Author &emsp;</sub> <br>
  </sup>Correspondence: zhangrongjunchen@myhexin.com <sup>
</p>

<p align="center">
  🤗<a href="https://huggingface.co/datasets/HiThink-Research/FinMTM">[Huggingface]</a>
</p>
<!-- <p align="center">
  <b>FinMTM</b> is a multi-turn multimodal benchmark that evaluates financial VLMs under three settings:
  objective questions, open-ended dialogues, and agent-based tasks with tool-use and multi-source evidence.
</p> -->
<br>
<p align="center">
  <img src="static/main2.png" hetight="320" />
</p>
<p align="center"><i>Overview of FinMTM: task types and capability coverage.</i></p>

---

## 🔥 Updates
- **2026-07**: Evaluation code and documentation synchronized with the submitted manuscript.
- **2026-01**: Initial release of benchmark dataset and paper.
- **TBD**: Online leaderboard opens for submissions.

---

## 📌 Contents
- [Overview](#-overview)
- [Representative Examples](#-representative-evaluation-examples)
- [Results](#-results)
- [Evaluation](#-evaluation)
- [Quickstart](#-quickstart)
- [License](#-license)
- [Citation](#-citation)

---

## 🧭 Overview
Financial reasoning is challenging for VLMs due to specialized chart formats, dense domain knowledge, long-horizon dependencies, and evidence-grounded tool use. Existing benchmarks are mostly single-turn and do not sufficiently measure **multi-turn dialogue stability**, **session-level memory**, or **agentic planning and execution**.

**FinMTM** addresses this gap by providing:
- **Objective questions**: single and multiple choice questions grounded in financial visuals.

- **Open-ended questions**: multi-turn conversations that stress compositional reasoning, multi-step calculation, self-correction, and memory.

- **Financial agent task**: tool-augmented multi-source workflows with long-horizon planning and evidence-grounded answers.


---
**Data Construction Pipeline**

We propose a novel multi-stage data construction pipeline to scale multi-turn financial sessions, ensuring alignment with targeted cognitive requirements and traceability to verifiable evidence.

<p align="center">
<img src="static/12d206da-5383-4200-9290-d43333931b24.png" alt="Fig. X. Multi-stage construction pipeline of FinMTM." style="zoom: 33%;" />
</p>

Our multi-stage construction pipeline. We progressively build (i) objective visual-grounded items, (ii) multi-turn open-ended sessions emphasizing composition/calculation/self-correction/memory, and (iii) agentic workflows with tool planning, tool execution, and evidence-grounded responses.

## 🧪 Representative Evaluation Examples

The following examples cover all four FinMTM capability levels. They are
grounded in authentic financial charts and research-report pages and illustrate
the reasoning requirements of the benchmark.

Because several tasks require visual estimation, numerically equivalent answers
within the stated tolerance ranges should be accepted when the reasoning is
consistent.

### Capability Overview

| Level | Capability | Primary evaluation focus |
|---|---|---|
| L1 | Visual comprehension and spatial grounding | Curve localization, temporal ordering, threshold comparison, and counterfactual transformation |
| L2 | Multi-step numerical reasoning | Dual-axis interpretation, growth calculation, cross-series comparison, and elasticity-based projection |
| L3 | Self-correction | Detection and correction of an erroneous Assistant response and its downstream implications |
| L4 | Multi-page memory | Cross-page evidence retrieval, financial derivation, accounting for scope differences, and evidence-aware synthesis |

---

### L1: Visual Comprehension and Spatial Grounding

![A multi-series financial chart used for the L1 example](https://huggingface.co/datasets/HiThink-Research/FinMTM/resolve/main/media/images/6e6b0962f0ae10ac97688533aabfbfc9faff2cd74269a6febcbbc96247d710f6_98e7afb65de156e2.jpg)

#### Evaluation Prompt

Focus on the green curve around January 2024.

1. Identify the local minimum around January 2024 as point \(A\), and the short plateau reached after the subsequent rebound as point \(B\). Estimate their vertical values and describe their temporal and spatial relationship.
2. Calculate the rebound from \(A\) to \(B\), both in absolute terms and as a percentage of \(|A|\).
3. Suppose the shock that produced point \(A\) is more persistent and only 75% of the observed rebound is realized. Calculate the resulting endpoint \(B'\).
4. Determine whether \(B'\) lies above \(-30\), and quantify the distance from that threshold.
5. Between point \(A\) and the next major time tick, 2024-02, does the green curve touch or fall below \(-40\) again? Support the answer with visual evidence.

#### Reference Answer

Point \(A\) is the green curve’s local minimum around January 2024, at approximately \(-50\). Point \(B\) appears later in time and above \(A\), at approximately \(-20\).

The rebound is:

\[
\Delta=B-A=(-20)-(-50)=30.
\]

Relative to the magnitude of \(A\), the recovery is:

\[
\frac{30}{50}\times100\%=60\%.
\]

If only 75% of the observed rebound is realized:

\[
B'=-50+30\times75\%=-27.5.
\]

Because \(-27.5>-30\), the adjusted endpoint is 2.5 units above the \(-30\) threshold:

\[
-27.5-(-30)=2.5.
\]

The green curve does not touch or fall below \(-40\) again before the 2024-02 tick. After the local minimum, it rebounds sharply and remains near the \(-20\) level over the relevant interval.

#### What This Example Tests

- Correct identification of the target curve among multiple series.
- Joint reasoning about horizontal time order and vertical position.
- Separation of rebound magnitude, recovery percentage, and endpoint value.
- Correct comparison of signed values against a negative threshold.
- Evidence-based verification within a specified temporal interval.

---

### L2: Multi-step Numerical Reasoning

![National and Ningbo securities-transfer balances shown on two vertical axes](https://huggingface.co/datasets/HiThink-Research/FinMTM/resolve/main/media/images/46fe1758aee952883339c672495e987128d10c69bbc6ebf276250055d3c03eb5_0b7008c654344591.jpg)

The blue series represents the national securities-transfer balance and uses the left vertical axis. The red series represents the Ningbo balance and uses the right vertical axis.

#### Evaluation Prompt

1. Estimate both series in June 2024 and June 2025. For the remaining calculations, use the following central estimates:
   - National balance: approximately RMB 1.70 trillion in June 2024 and RMB 2.80 trillion in June 2025.
   - Ningbo balance: approximately RMB 22.0 billion in June 2024 and RMB 26.0 billion in June 2025.
2. Calculate the one-year growth rate of each series and the percentage-point difference between them.
3. Define the growth elasticity of Ningbo relative to the national series as:

   \[
   E=\frac{\text{Ningbo growth rate}}{\text{national growth rate}}.
   \]

   Calculate \(E\).
4. Assume that the national balance grows by 12% in the following year and that the same elasticity is maintained. Estimate Ningbo’s corresponding growth rate and balance.
5. Normalize each June 2025 value by the maximum of its own vertical axis: RMB 4.00 trillion for the national series and RMB 50.0 billion for Ningbo. Which series is closer to the top of its own scale?

#### Reference Answer

Reasonable visual-estimation ranges are:

- National, June 2024: RMB 1.65–1.75 trillion.
- National, June 2025: RMB 2.75–2.85 trillion.
- Ningbo, June 2024: RMB 21.0–23.0 billion.
- Ningbo, June 2025: RMB 25.0–27.0 billion.

Using the central estimates, national growth is:

\[
g_{\text{national}}
=\frac{28{,}000-17{,}000}{17{,}000}\times100\%
\approx64.71\%.
\]

Ningbo growth is:

\[
g_{\text{Ningbo}}
=\frac{260-220}{220}\times100\%
\approx18.18\%.
\]

The difference is:

\[
64.71\%-18.18\%=46.53\text{ percentage points}.
\]

The estimated elasticity is:

\[
E=\frac{18.18\%}{64.71\%}\approx0.281.
\]

If the national balance subsequently grows by 12%, Ningbo’s implied growth rate is:

\[
12\%\times0.281\approx3.37\%.
\]

The projected Ningbo balance is therefore:

\[
260\times(1+3.37\%)\approx268.77
\]

or approximately RMB 26.88 billion.

After normalization by the corresponding axis maximum:

\[
\frac{28{,}000}{40{,}000}=70\%,\qquad
\frac{260}{500}=52\%.
\]

The national series is therefore closer to the top of its own scale. The apparent pixel heights of the two curves should not be compared directly because the chart uses two different vertical axes.

#### What This Example Tests

- Correct association of each curve with its vertical axis.
- Percentage-change calculations using the proper base period.
- Distinction between percentage changes and percentage-point differences.
- Multi-stage projection using an explicitly defined elasticity.
- Scale-aware comparison in a dual-axis chart.

---

### L3: Self-correction

![BDTI and BCTI indices across multiple years](https://huggingface.co/datasets/HiThink-Research/FinMTM/resolve/main/media/images/201a8700f26f8360976058a6118f6bcb7eee9d49e32c45019101b91b7a6b0ba0_481ad373872cbe20.jpg)

#### Conversation Context

**Turn 1 — User**

Compare the BDTI and BCTI values near the end of January in 2024 and 2023. Use the following approximate readings:

- BDTI: 1,300 in 2024 and 900 in 2023.
- BCTI: 1,350 in 2024 and 700 in 2023.

Calculate the year-over-year point difference for each index.

**Turn 1 — Assistant**

The BDTI difference is approximately \(1{,}300-900=400\) points. The BCTI difference is approximately \(1{,}350-700=650\) points. Both indices are higher in 2024 than in the corresponding period of 2023.

**Turn 2 — User**

Which index shows the stronger year-over-year improvement? Briefly explain.

**Turn 2 — Assistant response included in the evaluation context**

> BCTI decreased from approximately 1,350 in 2023 to approximately 700 in 2024, producing a year-over-year difference of \(-650\) points. The product-tanker market was therefore substantially weaker than one year earlier. By contrast, BDTI increased by approximately 400 points, indicating that only the crude-tanker market improved.

The Turn 2 Assistant response is intentionally incorrect. The evaluated model receives this response as part of the prior conversation and must identify and repair the error.

#### Evaluation Prompt

1. Identify the erroneous Assistant turn and specify which year-to-value mapping was reversed or altered.
2. Recalculate the correct year-over-year point differences for BDTI and BCTI.
3. Using the 2023 value as the base, calculate the relative year-over-year increase of each index and determine which one improved more strongly.
4. Conduct a stress test in which both 2024 readings are 15% lower than the values shown. Recalculate their changes relative to 2023 and determine whether both indices would still remain above their 2023 levels.
5. State the corrected market conclusion without carrying forward the claim that only the crude-tanker market improved.

#### Reference Answer

The error appears in the Turn 2 Assistant response. It reverses the BCTI year-to-value mapping. The correct mapping is approximately 1,350 in 2024 and 700 in 2023, not the reverse. This also changes the correct point difference from \(+650\) to the erroneous value of \(-650\).

The correct differences are:

\[
\Delta_{\text{BDTI}}=1{,}300-900=400\text{ points},
\]

\[
\Delta_{\text{BCTI}}=1{,}350-700=650\text{ points}.
\]

The relative increases are:

\[
g_{\text{BDTI}}
=\frac{400}{900}\times100\%
\approx44.44\%,
\]

\[
g_{\text{BCTI}}
=\frac{650}{700}\times100\%
\approx92.86\%.
\]

BCTI shows the stronger relative improvement.

Under the 15% stress scenario, the adjusted 2024 readings are:

\[
1{,}300\times85\%=1{,}105,
\qquad
1{,}350\times85\%=1{,}147.5.
\]

Their changes relative to 2023 are:

\[
\frac{1{,}105-900}{900}\times100\%
\approx22.78\%,
\]

\[
\frac{1{,}147.5-700}{700}\times100\%
\approx63.93\%.
\]

Both stressed values remain above their corresponding 2023 levels. The corrected conclusion is that both BDTI and BCTI improved year over year near the end of January, with BCTI showing the larger relative increase. The figure does not support the claim that only the crude-tanker market improved.

#### What This Example Tests

- Attribution of the error to an Assistant response rather than to the User.
- Correction of the source facts, numerical derivations, and downstream interpretation.
- Use of the corrected context in subsequent calculations.
- Re-evaluation of the conclusion under a counterfactual stress scenario.

---

### L4: Multi-page Memory

<p>
  <img src="https://huggingface.co/datasets/HiThink-Research/FinMTM/resolve/main/media/images/d283185552f09cdf5d81b1fddbdb916409f8ba5d7a444e8863ff84b31aa5c450_9332df7318e60025.jpg" width="19%" alt="Research report, page 1">
  <img src="https://huggingface.co/datasets/HiThink-Research/FinMTM/resolve/main/media/images/bc699052e39b62f6f60d0a701469e9f25551618e539f4f7a8effb9cd95f47372_022710b8e1460a71.jpg" width="19%" alt="Research report, page 2">
  <img src="https://huggingface.co/datasets/HiThink-Research/FinMTM/resolve/main/media/images/5d5f22ec69f0712b152d4ec87deefd5ad283d82f6ff559cfd01000dd80095a9a_2d07edca5766e2cc.jpg" width="19%" alt="Research report, page 3">
  <img src="https://huggingface.co/datasets/HiThink-Research/FinMTM/resolve/main/media/images/d456754ffaea6558ac7ec71cf2d285b3bb6d499fa7de06b1ac4ea1ef8ce37929_cfc48330563e8bfd.jpg" width="19%" alt="Research report, page 4">
  <img src="https://huggingface.co/datasets/HiThink-Research/FinMTM/resolve/main/media/images/caf24509560b6e3be5800be3bfa2a32c92f1d36578bc58b83296029e47d16d93_e3e20eb74d508625.jpg" width="19%" alt="Research report, page 5">
</p>

#### Evaluation Prompt

Assume the role of an investment-committee analyst and prepare an evidence-based review of the five-page report.

1. From page 1, extract first-quarter 2025 revenue, year-over-year revenue growth, and attributable net profit. From page 2, extract the full-year 2025 revenue and net-profit forecasts.
2. Calculate the proportion of forecast full-year revenue already generated in the first quarter and the revenue that must be generated over the remaining three quarters.
3. Assuming that quarterly attributable net profit and the annual net-profit forecast are approximately comparable in scope, calculate the net profit required over the remaining three quarters to meet the 2025 forecast. Derive the corresponding net margin on the remaining revenue and compare it with the 2024 actual and 2025 forecast full-year margins on page 2.
4. Calculate the forecast growth in revenue and net profit from 2024 to 2025. Determine whether earnings are expected to recover faster than revenue.
5. Page 1 reports a current share price of RMB 65.41 and a six-month target price of RMB 68.33. Calculate the implied absolute price return. Using the rating definition on page 4, assess whether this absolute return alone is sufficient to conclude that the “Overweight-A” rating is internally inconsistent.
6. Page 1 estimates peak sales of RMB 22.8 billion for the nine-valent HPV vaccine, while page 2 forecasts 2027 revenue of RMB 28.6346 billion. Calculate the ratio between these two figures and explain why it should not be reported as the vaccine’s 2027 revenue share.
7. Summarize at least three disclosed risks. In light of the disclaimer on page 5, separate the final assessment into disclosed facts, analyst forecasts or judgments, and independently derived calculations.

#### Reference Answer

##### Cross-page evidence retrieval

Disclosed first-quarter results on page 1:

- First-quarter 2025 revenue: RMB 401 million.
- Year-over-year revenue change: \(-46.76\%\).
- Attributable net profit: a loss of RMB 53 million.

Analyst forecasts on page 2:

- Full-year 2025 revenue: RMB 3,166.5 million.
- Full-year 2025 net profit: RMB 345.8 million.

##### Full-year revenue completion

\[
\text{Q1 revenue share}
=\frac{401}{3{,}166.5}\times100\%
\approx12.66\%.
\]

The remaining revenue requirement is:

\[
3{,}166.5-401
=2{,}765.5\text{ million yuan}.
\]

##### Implied profitability for the remaining three quarters

Because the first quarter recorded a loss of RMB 53 million, the remaining three quarters would need to generate:

\[
345.8-(-53)
=398.8\text{ million yuan}
\]

to meet the full-year forecast.

The corresponding margin on the remaining revenue is:

\[
\frac{398.8}{2{,}765.5}\times100\%
\approx14.42\%.
\]

Page 2 reports a 2024 actual net margin of 4.7% and a 2025 forecast margin of 10.9%. The implied 14.42% margin for the remaining three quarters exceeds both figures, indicating that the full-year target requires a substantial improvement in profitability after the first-quarter loss. This inference depends on the stated assumption that the quarterly and annual profit figures are sufficiently comparable in scope.

##### Revenue and earnings recovery

Forecast revenue growth from 2024 to 2025 is:

\[
\frac{3{,}166.5-2{,}245.4}{2{,}245.4}\times100\%
\approx41.02\%.
\]

Forecast net-profit growth is:

\[
\frac{345.8-106.2}{106.2}\times100\%
\approx225.61\%.
\]

Net profit is therefore forecast to recover substantially faster than revenue, implying a marked improvement in profitability.

##### Target price and rating definition

The target price implies an absolute price return of:

\[
\frac{68.33-65.41}{65.41}\times100\%
\approx4.46\%.
\]

Page 4 defines the “Overweight-A” category using the expected six-month return relative to the CSI 300 Index, whereas 4.46% is an absolute price return derived from the target price. The two measures are not directly equivalent. Total return may also include dividends. Therefore, the absolute target-price return alone is insufficient to establish an inconsistency, although the analyst should disclose the benchmark-return and total-return assumptions more explicitly.

##### Peak vaccine sales and 2027 revenue

\[
\frac{22.8}{28.6346}\times100\%
\approx79.62\%.
\]

This calculation only shows that the estimated peak vaccine sales equal approximately 79.62% of forecast 2027 company revenue. The report does not establish that peak vaccine sales will occur in 2027, nor that the figures use identical recognition periods or accounting scopes. The result should therefore not be described as the vaccine’s 2027 revenue share.

##### Risk and evidence classification

Disclosed risks include:

1. A slower-than-expected launch of the nine-valent HPV vaccine.
2. Lower-than-expected HPV vaccine market penetration.
3. Intensifying market competition.
4. Lower-than-expected product sales.
5. Slower-than-expected international expansion.

| Evidence class | Examples |
|---|---|
| Disclosed facts | First-quarter revenue of RMB 401 million, revenue decline of 46.76%, and attributable net loss of RMB 53 million |
| Analyst forecasts or judgments | 2025 financial forecasts, the RMB 68.33 target price, estimated peak vaccine sales of RMB 22.8 billion, and the “Overweight-A” rating |
| Independently derived calculations | First-quarter revenue completion of 12.66%, required remaining-period margin of 14.42%, implied target-price return of 4.46%, and peak-sales-to-2027-revenue ratio of 79.62% |

The page 5 disclaimer states that forecasts and opinions are not guarantees and that the report should not serve as the sole basis for an investment decision. Forecasts, target prices, and peak-sales estimates should therefore be treated as risk-bearing analyst judgments rather than established outcomes.

#### What This Example Tests

- Retention and integration of information across five report pages.
- Consistent handling of reported results, forecasts, and independently derived values.
- Recognition and disclosure of potential scope differences between financial measures.
- Distinction between absolute stock return and benchmark-relative rating criteria.
- Avoidance of unsupported temporal alignment between peak-sales and annual-revenue forecasts.
- Evidence-aware synthesis that respects the report’s stated risk and disclaimer boundaries.

---

## 📊 Results
We benchmark a range of 22 leading VLMs on FinMTM. The final score is the average across:
**Objective Questions**, **Open-Ended Questions**, and **Financial Agent**.

<p align="center">
  <img src="static/e1b91bdd-de1b-45eb-9dac-19a42f6f66f3.png" width="900" />
</p>
<p align="center"><i>Comparison of leading VLMs on FinMTM. Final score is the average of Objective, Open-Ended, and Agent tasks.</i></p>


### Benchmark Results


<details>
  <summary> Benchmark Results (Click to Expand)</summary>

  <br>

**Column Definitions**
- **Objective Questions:** Single-choice (Obj-Single), Multiple-choice (Obj-Multi)
- **Open-Ended Questions:** Comprehension (Open-Com.), Calculation (Open-Cal.), Self-Correlation (Open-SelfCorr.), Memory (Open-Mem.)
- **Financial Agent Tasks:** With fuzzing (Agent-w fuzz), Without fuzzing (Agent-w/o fuzz)

| Method | Obj-Single | Obj-Multi | Open-Com. | Open-Cal. | Open-SelfCorr. | Open-Mem. | Agent-w fuzz | Agent-w/o fuzz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| _Proprietary Models_ |||||||||
| ChatGPT-4o | 79.3 | 49.1 | 77.2 | 76.8 | 46.2 | 38.9 | 29.7 | 34.8 |
| ChatGPT-o3* | 85.8 | 73.3 | 83.8 | 78.6 | 52.8 | 43.6 | 31.4 | 35.2 |
| ChatGPT-5* | 89.0 | **79.6** | <u>86.9</u> | <u>80.7</u> | <u>56.9</u> | <u>46.7</u> | 35.9 | 49.7 |
| Gemini 3 Flash | <u>91.9</u> | 78.1 | 82.2 | 76.0 | 55.4 | 41.6 | **53.6** | **62.6** |
| Grok-4-fast-non-reasoning* | 71.0 | 46.8 | 66.0 | 61.2 | 39.9 | 24.8 | 30.2 | 39.7 |
| Gemini 3 Pro | **92.1** | <u>78.4</u> | **87.5** | **82.8** | **58.8** | **48.5** | <u>48.3</u> | <u>54.3</u> |
| _InternVL Series_ |||||||||
| InternVL2.5-8B | 63.8 | 25.7 | 55.1 | 49.2 | 26.5 | 16.7 | 8.4 | 10.5 |
| InternVL2.5-26B | 70.5 | 31.3 | 61.7 | 57.7 | 32.3 | 22.8 | 11.2 | 14.0 |
| InternVL2.5-40B | 72.3 | 35.2 | 66.1 | 64.6 | 36.2 | 26.7 | 13.5 | 16.8 |
| InternVL3-78B | 75.6 | 42.4 | 76.2 | 77.6 | 43.6 | 32.6 | 18.2 | 22.8 |
| _Other VL Series_ |||||||||
| MiMo-VL-7B | 61.1 | 21.4 | 75.1 | 75.4 | 47.2 | 39.9 | 20.2 | 25.5 |
| GLM4.5V-108B | 73.7 | 51.0 | 85.4 | 79.6 | 51.1 | 42.2 | 26.5 | 32.4 |
| _Qwen VL Series_ |||||||||
| Qwen2.5-VL-3B | 64.5 | 16.4 | 68.2 | 67.7 | 40.5 | 27.6 | 9.4 | 11.9 |
| Qwen2.5-VL-7B | 73.4 | 24.1 | 74.3 | 73.4 | 43.1 | 33.9 | 11.1 | 14.2 |
| Qwen3-VL-4B-Instruct | 73.3 | 34.2 | 74.5 | 71.2 | 39.5 | 25.9 | 15.1 | 19.1 |
| Qwen3-VL-4B-Thinking | 66.1 | 24.3 | 71.2 | 68.5 | 42.5 | 31.0 | 12.8 | 15.6 |
| Qwen3-VL-30B-A3B-Instruct | 77.2 | 47.3 | 82.1 | 76.5 | 42.5 | 33.7 | 16.2 | 20.8 |
| Qwen3-VL-30B-A3B-Thinking | 71.5 | 49.4 | 80.7 | 67.1 | 44.2 | 35.1 | 18.9 | 23.3 |
| Qwen3-VL-32B-Instruct | 84.5 | 39.9 | 84.3 | 80.7 | 50.8 | 40.3 | 19.6 | 25.1 |
| Qwen3-VL-32B-Thinking | 83.4 | 46.5 | 80.3 | 68.6 | 43.5 | 33.7 | 23.2 | 28.6 |
| Qwen3-VL-235B-A22B-Instruct | 81.3 | 48.5 | 85.5 | 80.9 | 54.5 | 41.5 | 32.1 | 38.7 |
| Qwen3-VL-235B-A22B-Thinking | 80.5 | 42.3 | 84.5 | 79.4 | 52.5 | 43.0 | 35.2 | 41.5 |

</details>

#### 💡 Key Observations
- Agentic settings expose larger gaps than pure reasoning-only settings.
- Removing identifiable entities increases difficulty and stresses evidence-grounded reasoning.
- Scaling helps, but robust tool planning and execution remain a major bottleneck for open-source models.

---


## 📏 Evaluation
FinMTM uses the paper-aligned, task-aware protocols implemented in
[`finmtm_eval/metrics.py`](finmtm_eval/metrics.py). LLM judges identify
semantic matches and assign rubric scores; all metric arithmetic is recomputed
deterministically in Python. See
[`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) for formulas,
scales, and input/output fields.

### 1) Objective Questions
- Equation (1): any over-selected option yields zero.
- Otherwise, the score is the fraction of gold options selected.
- Single choice is the special case with one gold option.

### 2) Open-Ended Dialogues (Multi-turn)
- Each turn is scored on VP, FL, DA, CMV, and TA (0-10 each) with uniform
  weights.
- The session is scored with its L1-L4 task-specific checklist (0-10).
- Equation (4) fixes turn/session weights at 0.5/0.5.
- Internal scores are 0-10; reported benchmark scores are scaled to 0-100.

### 3) Financial Agent Tasks
- Tool calls are matched as unordered sets by functional alignment and core
  arguments.
- Planning uses recall-oriented F-beta with `beta=2`, scaled to 0-25 as
  `25 * F2`; EMR is reported separately as an exact-match diagnostic.
- Answer correctness is graded on 0-50 (not binary); reasoning is 0-25.
- The final score is `answer + reasoning + tool`, on a 0-100 scale.

---


## ⚡ Quickstart

### 1. Environment Setup

Download the dataset from the
[Hugging Face dataset](https://huggingface.co/datasets/HiThink-Research/FinMTM).
The expected JSONL fields are documented in
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).

```bash
cd FinMTM-main
conda create -n finmtm_env python=3.10 -y
conda activate finmtm_env
pip install -r requirements.txt
```

### 2. Inference
#### 2.1 Inference for Objective Questions (Single/Multiple Choice)
```bash
python -m inference.SC_MC.etest \
  --input /path/to/objective.jsonl \
  --output outputs/objective_results.jsonl \
  --summary outputs/objective_summary.json \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct
```
#### 2.2 Inference for Multi-Turn QA
```bash
python -m inference.MTQA.inference \
  --input-dir /path/to/open_ended \
  --output-dir outputs/open_ended \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct
```
#### 2.3 Financial Agent

Agent inference requires an MCP server that implements the five tool interfaces
listed in [`docs/MCP_INTERFACE.md`](docs/MCP_INTERFACE.md).

```bash
python -m judge.Agent.main \
  --mode all \
  --input /path/to/agent.jsonl \
  --out-root outputs/agent \
  --api-base http://localhost:8000/v1 \
  --model Qwen3-VL-30B-A3B-Instruct \
  --judge-api-base http://localhost:8000/v1 \
  --judge-model Qwen3-VL-235B-A22B-Instruct \
  --mcp-url http://localhost:8081/sse
```
### 3. Evaluation

Open-ended dialogue evaluation:

```bash
python -m judge.MTQA.main \
  --dirs /path/to/open_ended_outputs \
  --client openai \
  --api_base http://localhost:8000/v1 \
  --model ChatGPT-4o
```

Evaluate an existing Agent trace without running MCP inference:

```bash
python -m judge.Agent.main \
  --mode eval \
  --input /path/to/trace.jsonl \
  --out-root outputs/agent_scores \
  --judge-api-base http://localhost:8000/v1 \
  --judge-model ChatGPT-4o
```

Run deterministic metric and aggregation tests:

```bash
python -m unittest discover -s tests -v
```

## 📄 License

![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg) ![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)

**Code:** Apache 2.0
**Dataset:** CC BY-NC 4.0
Research-use only. Must comply with: https://openai.com/policies/terms-of-use.

## 📚 Citation
If you find our work useful, please consider citing:

```bibtex
@misc{zhang2026finmtm,
      title={FinMTM: A Multi-Turn Multimodal Benchmark for Financial Reasoning and Agent Evaluation},
      author={Chenxi Zhang and Ziliang Gan and Liyun Zhu and Youwei Pang and Qing Zhang and Rongjunchen Zhang},
      year={2026},
      eprint={2602.03130},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.03130},
}
```
