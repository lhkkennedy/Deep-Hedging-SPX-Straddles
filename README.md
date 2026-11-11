# Deep Hedging SPX Straddles

### Simulation pretraining, market fine-tuning and maturity pooling under Heston and lifted Heston

**MSc Finance and Machine Learning dissertation | Queen Mary University of London | 2026**  
**Dissertation grade: 92/100** · Selected as the School of Economics and Finance's best dissertation for nomination to the **Worshipful Company of International Bankers Prize**

> **Research question:** Can neural hedging policies reduce the terminal risk of short S&P 500 straddles relative to conventional delta hedging when rebalancing is discrete, costly and constrained by information available at the decision time?

This project investigates **cost-aware, daily hedging of short, near-at-the-money S&P 500 index (SPX) straddles**. It combines stochastic-volatility simulation, neural-network training, historical option-market evaluation and a controlled comparison of three design choices:

1. **Training distribution:** market-only learning versus pretraining on Black–Scholes, Heston or lifted-Heston paths.
2. **Market adaptation:** no fine-tuning versus output-layer or full-network fine-tuning after simulation pretraining.
3. **Sharing information across maturities:** a dedicated one-month model, a fully shared model, or shared hidden layers with maturity-specific output heads.

The main result is a **16.3% reduction in premium-normalised mean squared hedging error** for the leading neural policy compared with Black–Scholes delta at an assumed **1 basis point (bp) of proportional underlying trading cost**. The result is economically nuanced: **average P&L worsens, turnover and trading costs increase, and the improvement is not statistically confirmed across annual evaluation periods**. The work therefore identifies a potential form of **tail-error compression at a cost**, not a demonstrated improvement in risk-adjusted trading profitability.

**Scope:** This README describes the *final submitted dissertation* and the recovered July–August 2026 research artifacts. It supersedes descriptions of earlier, simulation-only and intraday experiments. The public repository's existing large notebook is an earlier research iteration; it should not be treated as the implementation supporting the final figures until the final notebooks are curated and added. **This is a research archive, not a production trading system or a one-command reproducibility package.**

---

## Key results

The leading policy is a **fully pooled, lifted-Heston-pretrained neural hedge, without subsequent market fine-tuning**. Its risk metrics are measured over historical out-of-sample straddle episodes using the same terminal accounting rules as the benchmark.

| Metric at 1 bp stock-trading cost | Leading neural hedge | Black–Scholes delta |
|---|---:|---:|
| Premium-normalised MSE ↓ | **0.154984** | 0.185212 |
| Premium-normalised RMSE ↓ | **0.393680** | 0.430362 |
| Premium-normalised expected shortfall, ES95 ↓ | **0.780805** | 0.840251 |
| Mean stock turnover per episode ↓ | 3.573 | **3.428** |
| Mean realised stock-trading cost (SPX points) ↓ | 3.842 | **3.794** |

**Interpretation:** Lower MSE does **not** mean 16.3% higher returns, and it is not a 16.3% reduction in RMSE. MSE penalises large errors quadratically. In the dissertation's paired accounting decomposition, the leading policy has approximately **4.765 SPX points lower mean terminal P&L** than Black–Scholes delta. The additional commissions are comparatively small; different stock-hedging gains explain most of that P&L gap.

### Where did the reduction in squared error come from?

The pooled episode-level decomposition of the neural-minus-Black–Scholes MSE difference shows:

| Component | Contribution to MSE difference |
|---|---:|
| Reduction in hedging-error variance | −0.030451 |
| Increase in squared bias | +0.002023 |
| **Net reduction in MSE** | **−0.028428** |

Nearly all the improvement is attributable to reducing large squared errors. The worst **5% of episodes ranked by baseline squared error** contribute approximately **−0.030599** to the MSE difference, while the remaining 95% contribute **+0.002172**. The effect resembles paying a small performance cost in ordinary conditions to compress unusually large *absolute* hedging errors; it does **not** by itself demonstrate protection against one-sided downside losses.

The improvement was also concentrated in specific market periods: **2018 and 2020 account for 92% of the summed nine-year MSE improvement**. The annual-weighted MSE difference had an unadjusted 95% interval of approximately **[−0.066849, +0.000026]**, which includes zero. Results are selected from an evaluated model grid and should not be described as independently confirmed superiority.

### Other findings

- **Simulation transfer:** On average, successfully trained Heston/lifted-Heston pretrained policies had lower MSE than market-only policies, but the estimated difference was imprecise. Additional fine-tuning of the output layer or full network changed MSE very little.
- **Model complexity:** Lifted Heston provided the leading policy, but its average advantage over Heston was small and uncertain. A more expressive volatility model is not automatically a better hedge.
- **Maturity pooling:** On a *matched one-month evaluation sample*, parameter sharing improved the simulation-pretrained policies in several specifications. In contrast, the **market-only, one-month-specific** policy performed better than its pooled alternatives. Pooling creates a bias–variance trade-off, not a guaranteed gain from more observations.
- **Costs and horizon:** The leading policy's share of episodes with lower squared error fell as assumed trading costs increased, particularly for shorter-dated options. Results depend on maturity, trade frequency and the risk objective.

These findings are specific to the tested loss function, model class, sample, assumptions and benchmark implementations.

---

## Experimental design

| Dimension | Final dissertation specification |
|---|---|
| Liability | One **short near-ATM SPX call–put straddle** per episode |
| Hedge | Dynamic stock-equivalent inventory; the option liability is not traded after entry |
| Entry maturities | **1W, 2W, 1M, 2M, 3M** |
| Market inputs | Historical SPX option quotes, index observations, rates/dividends and derived implied-volatility-surface features |
| Policy information | Observable contract, return, surface, previous-inventory, time and cost variables; **no latent simulator variance/factors** |
| Decision/execution timing | Observe close at **t**, submit the hedge decision and execute at the next recorded close **t+1**; option and initial hedge enter together under the final timing convention |
| Training sources | Black–Scholes, Heston, lifted-Heston simulation; historical-market-only control |
| Neural variants | **27 simulation-source × fine-tuning × pooling combinations**, plus **3 market-only** pooling variants |
| Neural policy | Bounded residual correction to a Black–Scholes-delta anchor; current-state MLP family in the main factorial comparison |
| Pooling | Dedicated **1M** estimator; five-maturity shared estimator; shared representation with maturity-specific output heads |
| Selection and testing | Chronological walk-forward protocol; **18-month trailing market training** with subsequent annual evaluation and support thresholds |
| Out-of-sample period | **2016–2024**, nine eligible annual evaluation years |
| Evaluation | **2,030 straddle episodes** across **462 entry dates**; episode-level accounting and year-aware inference |
| Primary objective | Premium-normalised terminal **MSE** |
| Secondary outcomes | RMSE, MAE, bias, ES95, turnover, trading costs, mean P&L, maturity/cost/regime sensitivity |
| Costs | **1 bp primary**, with **5 and 10 bp** robustness comparisons |

The final comparison uses common annual test windows and underlying data contracts. Because a dedicated 1M model is evaluated only on 1M liabilities, **pooling methods must be compared on their shared 1M episode panel**; an all-maturity pooled score is not directly comparable to a 1M-specific score.

### Why the execution convention matters

This is a **daily close-to-close** study, not high-frequency trading. A decision based on the closing option surface at date `t` cannot be credited with a same-close execution after observing that close. The final pipeline uses a next-session execution proxy consistently in simulation, market training, validation, benchmarks and reported P&L.

The **SPX index close is not directly tradable**. Stock-hedge P&L calculated using SPX levels is an economic proxy, not evidence of executable futures or ETF fills. A deployment study would need timestamped SPY/futures prices, basis, financing, margin, liquidity and market impact.

---

## Research pipeline and source notebooks

The dissertation involved more than its final hedging notebook. The recovered July–August notebooks cover **market data, calibration, simulation, timing reconstruction, policy evaluation and forensic/financial diagnostics**.

The titles below are **curation targets from the recovered research archive**. They are *not* an assertion that every notebook is already committed to this repository or that the files can be run consecutively without adapting their cache and data contracts.

| Stage | Recovered artifact | Role and release status |
|---|---|---|
| 1. Empirical panel and calibration | `01_calibration_full_integrated_H035_H025.ipynb` | Integrated Heston/lifted-Heston calibration, surface diagnostics and earlier evaluation; **executed historical artifact** |
| 2. Timing reconstruction | `02_timing_correct_1m_banks.ipynb` | Rebuilds 1M observable episode banks without recalibration; **saved copy has no execution outputs** |
| 3. Five-maturity simulation | `03_build_h035_5bucket_simulations.ipynb` | Builds Black–Scholes/Heston/lifted-Heston close-to-close episodes for five maturity buckets; **saved copy has no execution outputs** |
| 4. Model-delta audit | `04_forensic_heston_lifted_delta_audit.ipynb` | Diagnoses maturity-unit, execution, model-state and calibration issues in model-based deltas; **executed diagnostic artifact** |
| 5. Final hedging experiment | `05_final_factorial_user_august.ipynb` | User-retained August final factorial experiment and exported results; **executed research artifact** |
| 6. Financial/statistical diagnostics | `06_august_diagnostic_review.ipynb` | Reporting-only plots and checks from locked results; **executed; does not refit models** |
| 7. Subsequent review | `07_august_review_implemented.ipynb` | Later reviewed implementation for version comparison; **not assumed automatically authoritative** |

The original July notebook additionally includes volatility-surface preparation, forward-variance fitting, calibration-diagnostic plots, parameter-stability analysis and simulation checks. The separate diagnostic review covers terminal-loss exceedance curves, turnover, P&L decomposition, regimes, cost sensitivity, training-seed stability and calendar/cohort support.

**Provenance:** The table records recovered versions, not a validated merged software release. Some notebooks duplicate upstream code, contain historical design revisions, or require locally cached calibration outputs. In particular, **H = 0.25 and H = 0.35 experiments must not be mixed**. The submitted headline comparison belongs to the final **H = 0.35** evidence set.

### A relevant forensic caveat

A separate July audit found a **maturity-unit error in an earlier Heston model-delta implementation** and additional state-alignment concerns for lifted-Heston delta. Corrected and uncorrected model-delta results must not be conflated. The headline **neural versus Black–Scholes** numbers above come from the dissertation's reported factorial comparison; that comparison does **not** depend on a claim that Heston or lifted-Heston delta was implemented flawlessly. The model-based baselines require independent reconciliation before strong conclusions are drawn from their relative ranking.

---

## Reproduction and data availability

### What is available

The archived notebooks contain the model definitions, data-processing logic, simulation and training protocols, evaluation code, diagnostic procedures and some **saved execution outputs**. They provide inspectable methodological evidence.

### What is not currently supplied

- **Licensed raw market data:** OptionMetrics IvyDB US and CRSP inputs are not redistributed.
- **All upstream calibration caches, simulation banks and fitted checkpoints:** Some notebooks rely on persisted local files absent from a clean checkout.
- **A fully validated, from-scratch build:** The recovered notebooks have not all been rerun together under a single pinned environment. The five-bucket bank builder and timing-rebuild copies have no saved execution outputs.
- **A deployable execution backtest:** Next-close SPX accounting is not a live stock/futures fill study.

A full replication requires lawful access to the underlying market datasets, restoration or regeneration of the calibrated five-maturity panels, and verification of the exact frozen experiment signatures and output schemas. **Do not treat stored notebook figures as proof that a clean rerun has succeeded.**

### Environment guidance

Use **Python 3.10+** and an isolated environment. The computational notebooks use standard numerical/data-science libraries including NumPy, pandas, SciPy, matplotlib, PyTorch and Parquet IO. Exact environment versions, device-specific requirements and memory constraints still need to be extracted and locked from the selected final notebooks.

```bash
python -m venv .venv
# macOS/Linux: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**Important:** The repository's existing `requirements.txt` was written for an earlier calibration notebook and **does not install PyTorch**. The command above is therefore an environment starting point, **not a verified complete setup for the final factorial experiment**. A reconciled, pinned requirements or environment lock file is a publication task.

### Data and privacy

Raw OptionMetrics, CRSP/WRDS extracts and reconstructed proprietary quote panels should not be committed or distributed without checking their licence terms. Notebook outputs, embedded tables, checkpoint files, Git history and cached artifacts also require inspection; `.gitignore` does not protect data already committed. The public release should contain only code, original documentation and permissible derived summaries or figures.

---

## Limitations and interpretation

1. **Objective choice matters.** The main objective is symmetric, zero-target squared terminal error. It penalises unusually large gains as well as losses, and does not directly maximise wealth or optimise one-sided capital risk. Utility, CVaR and capital-constrained objectives could rank policies differently.
2. **Statistical evidence is limited.** Only nine annual periods supply largely independent market environments; large improvements were concentrated in 2018 and 2020. The leading policy was chosen among tested variants, and the reported interval crosses zero.
3. **Simulation transfer is conditional.** Richer latent volatility dynamics need to produce useful *observable* option surfaces and return histories. Calibration fit, model delta accuracy and policy hedge value are distinct questions.
4. **Partial pooling can help or hurt.** Shared parameters reduce estimation variance when maturity structures are related, but can introduce bias or negative transfer when the training distribution does not support a common mapping.
5. **Some optimisation runs failed.** The dissertation flags unsuccessful Black–Scholes-pretrained runs; they cannot be treated as successful evidence about fine-tuning or pooling without retraining and diagnostics.
6. **Execution is stylised.** The study hedges with an index-close proxy at daily frequency, without timestamped implementable fills, exchange fees, funding/margin treatment or full market-impact modelling.
7. **The archive is not yet a production-quality package.** The recovered stages need explicit data licences, dependency pinning, executable smoke tests, cache-provenance checks and independent replication before being advertised as fully reproducible.

**Conclusion:** The central contribution is not a general claim that neural networks outperform delta hedging. It is an empirical decomposition of **which financial risk a learned hedge reduces, what economic performance it sacrifices, and how the training distribution and maturity sharing affect the result**.

---

## Context and references

The project builds on derivatives hedging, stochastic volatility, simulation-based learning and empirical deep hedging. Selected references from the dissertation include:

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
- Heston, S. (1993). *A Closed-Form Solution for Options with Stochastic Volatility.* Review of Financial Studies.
- Whalley, A. & Wilmott, P. (1997). *An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs.* Mathematical Finance.
- Bühler, H., Gonon, L., Teichmann, J. & Wood, B. (2019). [*Deep Hedging.*](https://doi.org/10.1080/14697688.2019.1571683) Quantitative Finance.
- Abi Jaber, E. (2019). *Lifting the Heston Model.* Quantitative Finance.
- Wiese, M., Bai, L., Wood, B. & Buehler, H. (2019). [*Deep Hedging: Learning to Simulate Equity Option Markets.*](https://arxiv.org/abs/1911.01700)
- Mikkilä, O. & Kanniainen, J. (2023). *Empirical Deep Hedging.* Quantitative Finance.
- Abi Jaber, E. & Li, S. (2025). *Volatility Models in Practice: Rough, Path-dependent or Markovian?* Mathematical Finance.

### Related ongoing research

Following the dissertation, the research programme has expanded to **stronger regression and nonlinear baselines, exotic-payoff hedging, simulation-to-market transfer, portfolio-level stochastic control, model predictive control (MPC), continuation-value learning and policy distillation**. These experiments are **subsequent exploratory work**, use different designs and benchmarks, and **are not evidence for the dissertation's 16.3% result**. They should be documented in a separate, clearly versioned research repository.

---

## Author and acknowledgement

**Lewis Hikari Kawase Kennedy**  
MSc Finance and Machine Learning · Queen Mary University of London

Dissertation supervision: **Dr Katerina Panopoulou**. Thanks also to dissertation teaching assistant **Denys Iliasov** and the School of Economics and Finance.

*Academic research only; not investment advice.*