# Paper Review: "Tests for Structural Breaks in Time Series Analysis: A Review of Recent Development"

**Course:** Data Analysis in Finance
**Authors of this review:** Pikuleva, Tolmacheva, Beznosenkova, Tsurkan

| | |
|---|---|
| **Authors** | P. Muthuramu (Rajiv Gandhi National Institute of Youth Development, Chennai), T. Uma Maheswari (Lady Doak College, Madurai) |
| **Journal** | Shanlax International Journal of Economics, vol. 7, no. 4, 2019, pp. 66–79 |
| **DOI** | [10.34293/economics.v7i4.628](https://doi.org/10.34293/economics.v7i4.628) |
| **Type** | Review article, open access, CC BY-SA 4.0 licence |
| **Keywords** | structural breaks, Chow test, Quandt–Andrews test, Bai–Perron test |

---

## 1. What the paper is about and why it matters

A **structural break** (or change point) is a moment after which the parameters of a time series model change: the mean, the slope, the variance. If a break exists and the model ignores it, the estimates become an "average" of regimes that never existed in reality, forecasts are biased, and unit root and cointegration tests give misleading results.

This is a central problem in finance: a stock's beta changes as the company's business model changes, the level of interest rates changes with the monetary policy regime, and volatility differs between calm and crisis periods. Any model calibrated on a long history (CAPM, VaR, an interest rate model, a trading strategy) implicitly assumes parameter stability.

The authors aim to **systematise the development of structural break tests from 1960 to the 2010s** and divide the literature into three generations:

1. **Tests for a known break date** — Chow (1960) and its generalisations.
2. **Tests for a single break at an unknown date** — Quandt (1960), CUSUM (Brown, Durbin & Evans, 1975), Andrews (1993), Andrews & Ploberger (1994).
3. **Tests for multiple breaks at unknown dates** — Bai (1997), Bai & Perron (1998, 2003, 2006).

The main thesis: the focus of the literature has shifted from testing a break *known in advance* to *searching for several unknown breaks* using dynamic programming.

## 2. Structure of the paper

| Section | Pages | Content |
|---|---|---|
| Introduction | 66–67 | History of the question, overview of extensions of the Chow test, outline of the paper |
| Test with Known Breakpoints | 67–68 | Chow test: analysis-of-variance (n₁, n₂ > k) and predictive (n₂ < k) versions; the heteroskedasticity problem; Hansen's (2001) example |
| Test of Unknown Structural Breaks | 68–71 | Quandt (QLR), Farley–Hinich, Kim–Siegmund, CUSUM and CUSUM², fluctuation test, Andrews sup-F, Andrews–Ploberger exp/ave-F |
| Multiple Unknown Structural Breaks | 71–77 | The Bai–Perron model, dynamic programming (Figure 1, Table 1), UDmax/WDmax, sequential test, information criteria, trimming, assumptions on the errors, the repartition procedure |
| Bai–Perron limitations, Discussion | 77–78 | Sensitivity of results to settings, final comparison of methods |

## 3. Detailed review of the content

### 3.1. Break at a known date: the Chow test

Consider two regressions for two periods
$$y_1 = X_1\beta_1 + \varepsilon_1, \qquad y_2 = X_2\beta_2 + \varepsilon_2,$$
with null hypothesis $H_0:\ \beta_1 = \beta_2$. Following Maddala (1998), the authors distinguish three forms of the test:

* **Analysis-of-variance test**, when both subsamples have more observations than parameters ($n_1, n_2 > k$):
$$F = \frac{(SSR_R - SSR_1 - SSR_2)/k}{(SSR_1+SSR_2)/(T-2k)} \sim F(k,\ T-2k).$$
* **Predictive test** ($n_2 < k$): it tests not the equality of coefficients but the unbiasedness of the forecast for the second subsample:
$$F = \frac{(SSR_R - SSR_1)/n_2}{SSR_1/(n_1-k)} \sim F(n_2,\ n_1-k).$$
* **Generalisation to J regimes** (Dufour, 1982), including the case where some subsamples are shorter than $k$.

The paper then lists extensions: to systems of simultaneous equations (Lo & Newey, 1985), to non-linear models with Wald/LM/LR versions (Andrews & Fair, 1988), to models with a lagged dependent variable (Ploberger, Krämer & Alt), and to autocorrelated errors.

An important point of this section is that **the Chow test assumes equal error variances across regimes**. Toyoda (1974) showed that under heteroskedasticity and unequal subsample sizes the actual significance level differs from the nominal one; Schmidt & Sickles (1977) refined this result. This directly concerns financial data, where volatility changes over time.

The second key limitation is illustrated with **Hansen's (2001)** example (US labour productivity, 1947–2001): the break date has to be chosen either *arbitrarily* (then the test may show nothing or show a break "in the wrong place") or *from the data* (then it is correlated with the sample and the standard F distribution is wrong). Hansen obtained different answers for 1973 and 1975 and therefore moved on to tests with an unknown date.

### 3.2. A single break at an unknown date

**Quandt (1960) — QLR / "Max Chow".** Compute the Chow statistic (more precisely, the likelihood ratio) for every possible date $\tau$ and take the maximum:
$$QLR = \max_{\tau \in [\pi_0 T,\ (1-\pi_0)T]} F(\tau).$$
The distribution of the maximum is non-standard: the critical values are higher than for F (because of the search over dates). The ends of the sample are trimmed (**trimming**), usually $\pi_0 = 15\%$, otherwise one of the regimes has too few observations. The authors describe the likelihood function of the regime-switching model and the problem that, because the date is discrete, the standard regularity conditions fail (Poirier, 1976).

**Farley & Hinich (1970)** — a test for a "gradual" break: $y_t = x_t(\beta_1 + \delta t) + \varepsilon_t$ (a break as a trend in the coefficients), with the break date equally likely at any point.

**Kim & Siegmund (1989)** — a likelihood ratio test for a "broken-line" regression where only the intercept, or both intercept and slope, change; they also give confidence regions for the date.

**CUSUM and CUSUM² (Brown, Durbin & Evans, 1975)** are built on **recursive residuals** — one-step-ahead forecast errors from a model estimated on the preceding observations:
$$w_t = \frac{y_t - x_t'\hat\beta_{t-1}}{\sqrt{1 + x_t'(X_{t-1}'X_{t-1})^{-1}x_t}}, \qquad CUSUM_r = \sum_{t=k+1}^{r}\frac{w_t}{\hat\sigma_w}.$$
Under stability the CUSUM path fluctuates around zero; after a break the residuals tend to have the same sign and the path crosses the linear boundaries $\pm a\,[\sqrt{T-k} + 2(r-k)/\sqrt{T-k}]$. CUSUM² uses the normalised sum of squared $w_t$ and reacts to changes in variance. Ploberger & Krämer developed the theory further: OLS-CUSUM based on ordinary residuals, local power, and the fluctuation test. The authors note that the power of CUSUM is limited.

**Andrews (1993)** derived the asymptotic distribution of sup statistics (sup-Wald, sup-LM, sup-LR) for one break at an unknown date in a general class of models (including GMM) and published critical values. **Andrews & Ploberger (1994)** showed that the tests that are optimal in terms of weighted average power are not the sup tests but the **exp** and **ave** statistics:
$$\text{exp-}W = \ln\Big(\frac{1}{N}\sum_\tau e^{W(\tau)/2}\Big), \qquad \text{ave-}W = \frac{1}{N}\sum_\tau W(\tau).$$
In the Discussion the combination of the two approaches is called the **Quandt–Andrews test** — this is what EViews and other packages implement.

### 3.3. Multiple breaks at unknown dates: Bai–Perron

**The model** with $m$ breaks and $m+1$ regimes:
$$y_t = x_t'\beta + z_t'\delta_j + u_t,\qquad t = T_{j-1}+1,\dots,T_j,\quad j = 1,\dots,m+1.$$
If $\beta$ does not change, this is a partial structural change model; if $p=0$ (all coefficients change), it is a pure structural change model.

**Estimation** is a global minimisation of the sum of squared residuals over all admissible partitions:
$$(\hat T_1,\dots,\hat T_m) = \arg\min_{T_1,\dots,T_m} S_T(T_1,\dots,T_m),\qquad T_i - T_{i-1} \ge h.$$
Full enumeration costs $O(T^m)$. Bai & Perron (2003) use **dynamic programming**: first the SSR of every admissible segment is computed once (at most $T(T+1)/2$ of them — this is the **triangular matrix** in Figure 1 of the paper), and then a Bellman recursion is solved in $O(T^2)$ for any $m$. Table 1 of the paper counts how many segments are excluded because of the minimum length $h$ (xa), lack of room for the following segments (xb) and for the preceding segments (xc).

**Tests:**

| Test | $H_0$ | $H_1$ |
|---|---|---|
| sup-F(k) | no breaks | exactly $k$ breaks |
| UDmax $= \max_{1\le m\le M} \text{sup-}F(m)$ | no breaks | 1 to $M$ breaks (equal weights) |
| WDmax | no breaks | 1 to $M$ (weights $c(q,\alpha,1)/c(q,\alpha,m)$ equalise the p-values) |
| sup-F(l+1 \| l) | $l$ breaks | $l+1$ breaks |

Bai & Perron (2006) **recommend** the following strategy: first UDmax/WDmax, to make sure that there are any breaks at all, then the sequential $l$ vs $l+1$ test to determine their number. An alternative is **information criteria**: BIC (Yao, 1988) and LWZ (Liu, Wu & Zidek, 1997), which has a stricter penalty. According to the BP simulations, BIC works well without autocorrelation, LWZ is reliable under $H_0$ but underestimates the number of breaks when they exist, and AIC strongly overestimates.

**Trimming** $\varepsilon = h/T$ determines the maximum number of breaks (BP tabulate $M$ = 9, 8, 5, 3, 2 for $\varepsilon$ = 0.05, 0.10, 0.15, 0.20, 0.25). With autocorrelated and heteroskedastic errors a larger trimming (0.15–0.20) is needed.

**Assumptions on the errors.** The paper lists the options of the Bai–Perron program: `cor_u` (autocorrelated errors), `het_u` (different error variances across regimes), `het_z` (different distributions of regressors across regimes). They determine the covariance matrix (HAC with the Quadratic Spectral kernel and Andrews bandwidth) and the confidence intervals for the dates.

**Confidence intervals for the dates** are built from an asymptotic framework in which the magnitude of the break shrinks to zero (Bai, 1997).

**The repartition procedure** (Bai, 1997): dates found sequentially one at a time are re-estimated with the other dates held fixed, which improves efficiency.

**Limitations of Bai–Perron** stressed by the authors (with reference to Dholakia & Sapre, 2011): the number and dates of breaks depend strongly on the chosen test, the assumed number of breaks, the trimming, the length of the series and even on whether nominal or real data are used. A sensitivity analysis is therefore needed.

### 3.4. The authors' conclusions

* The Chow test is the standard for a known date; in its dummy-variable form it is easy to test intercept and slope shifts separately.
* Quandt–Andrews solves the unknown-date problem, but only for one break.
* Only Bai–Perron provides a theoretically complete framework for several unknown breaks; practical implementation is made easier by the Bai–Perron (2003) programs and the R package `strucchange` (Zeileis et al.).
* Because results are sensitive, the authors recommend applying **both Bai–Perron and Quandt–Andrews** and comparing the results.

## 4. Critical assessment

### 4.1. Strengths

* **A useful classification.** The split into "known date / one unknown date / several unknown dates" gives a clear logic of how the methods developed and is convenient for choosing a test in practice.
* **Broad coverage of the classics** from 1960 to 2006: from Chow and Quandt to Bai–Perron, including less well-known work (Farley–Hinich, Kim–Siegmund, the fluctuation test).
* **Practical recommendations** from Bai & Perron (2006): which trimming to use for which error properties, which criterion to choose, in what order to apply the tests.
* **An honest section on limitations**: the paper does not sell Bai–Perron as a "silver bullet" and recommends sensitivity analysis.
* The illustration of **dynamic programming** with the triangular matrix is the clearest part of the paper.

### 4.2. Weaknesses and errors

**Substantive.**

1. **"Recent development" is not about 2019.** The most recent methodological work cited is Bai & Perron (2006), and the most recent reference overall is from 2011. Missing are: tests for breaks in **variance** (Inclán & Tiao, 1994 — ICSS, key for finance), breaks in cointegration (Gregory & Hansen, 1996), unit root tests with endogenous breaks (Zivot–Andrews is in the reference list but not discussed in the text; Lumsdaine–Papell, Lee–Strazicich), Markov switching models (Hamilton, 1989), PELT/binary segmentation algorithms, Bayesian methods, multivariate (Qu & Perron, 2007) and panel breaks, and real-time monitoring (Chu, Stinchcombe & White, 1996).
2. **No empirical analysis or comparison of tests of its own.** Properties (power, size) are described in words, with no tables or simulations. The claim that "if the date is known, Chow is more powerful" remains unquantified.
3. **Inaccuracies in the description of methods:**
   * The Quandt–Andrews test is said to "have the advantage of accounting for the non-linearity of models compared with OLS" — this is a confusion: sup-F is applied to OLS estimates; the general class of models (GMM) is a feature of Andrews' (1993) theory, not of the test as such.
   * "The QLR critical value gives a wider distribution than F and applies to large samples" — the point is different: the maximum over dates has a heavier right tail, so the critical values are **higher** and the F tables cannot be used.
   * The partial structural change model is described as one "where all parameters were constant" — the opposite: in a partial model **some** parameters are constant.
   * The link between trimming and the number of breaks is stated inconsistently: on pp. 74 and 76 "with ε = 0.05, 0.10, 0.15 five breaks are allowed", while further down p. 76 "with 0.10 the maximum is 8". In Bai & Perron (2003): 9, 8, 5, 3, 2 breaks for ε = 0.05, 0.10, 0.15, 0.20, 0.25.
   * In the CUSUM² section the expected value of the statistic, $(r-k)/(T-k)$, is denoted as "$\beta$ under the null hypothesis", which is confusing.
   * The key weakness of CUSUM is not explained: it has almost no power against shifts in coefficients of regressors with zero mean (Krämer, Ploberger & Alt, 1988). This is critical for finance — see our CAPM beta example.

**Arithmetic and technical errors.**

4. **Table 1 (pp. 73–74), second example ($T=53$, $h=8$, $m=5$):** the paper prints $xc = 371 - 270 - 49 - 28 = 24$, but $5\cdot 8\cdot 7 = 280$, so the formula gives $xc = 14$; the total "117 admissible segments" does not follow even from the printed numbers ($1431 - 350 - 960 - 24 = 97$). The paper's own formulas give 107, and a direct count using the Figure 1 rule gives 111 (checked in our code, `notebooks/analysis.ipynb`, Section 5.1).
5. **Formulas are partly unreadable**: equation (6) (Quandt's log-likelihood) and the SSR formula on p. 72 are garbled by typesetting; notation is mixed in equation (3); "Wald" is spelled "Walt" in several places.
6. **Duplicated text**: the paragraphs on WDmax and the sequential test on pp. 74 and 76 are almost word-for-word repetitions.
7. **Reference errors**: Quandt (1960) in the text, but Quandt (1958) in the list; Bai & Perron (1998, Econometrica) and Bai & Perron (2003, JAE) are listed with a single author, "Jushan Bai"; misspelled names (Karmer → Krämer, Farely → Farley, Lurnsdaine → Lumsdaine, Glyssels → Ghysels, Canterll → Cantrell, Ploeger → Ploberger); many cited works (Bai & Perron 1994, 2006; Perron 2005, 2006; Yao 1988; Liu, Wu & Zidek 1997; Andrews & Fair 1988; Toyoda 1974) are missing from the reference list.
8. **Language**: in places the meaning is hard to follow ("multiple strange breaks", "number of tears", "knot all", "when decorating is equal to .25") — this looks like automated paraphrasing and reduces the paper's value as a teaching review.

### 4.3. Overall assessment

The paper is useful as a **map of the classical methods** and as an introduction to the logic of Bai–Perron, but not as a stand-alone source: formulas and details are better taken from the original papers (Andrews, 1993; Bai & Perron, 1998, 2003; Perron, 2006 — the survey in the Palgrave Handbook) or from the textbook by Maddala & Kim (1998). The promise of "recent development" is not fulfilled, and the technical errors point to weak peer review.

**Our rating:** 6/10 — good structure and choice of key methods, weak execution.

## 5. Relevance for financial data analysis

| Task | Where breaks arise | Suitable test |
|---|---|---|
| Beta estimation / CAPM, hedging | change in business model, crises, index changes | Quandt–Andrews, Bai–Perron with HAC; **not** CUSUM |
| Interest rate models | change in monetary policy regime | Bai–Perron (mean shifts with HAC) |
| Risk management, VaR | volatility regimes | Bai–Perron on log RV, CUSUM², ICSS (Inclán–Tiao) |
| Pairs trading, cointegration | breakdown of a long-run relationship | Gregory–Hansen (not covered in the paper) |
| Strategy backtesting | "alpha" disappears after publication or a crisis | Chow at a known date (publication date), sup-F |
| Unit root tests | a level shift mimics non-stationarity (Perron, 1989) | Zivot–Andrews, Perron |

Practical rules that follow from the paper and our calculations:

1. If the date is **genuinely exogenous** (a regulator's decision, a change in trading rules), use the Chow test, but in its robust (HAC) version.
2. If the date was chosen "by looking at the chart", it is already endogenous, and sup-F is needed.
3. If there may be several breaks, use Bai–Perron: UDmax/WDmax → sequential test → comparison with BIC/LWZ → check with several values of ε.
4. For returns and volatilities, always account for heteroskedasticity and autocorrelation.

## 6. Empirical check (our code)

We implemented all the reviewed tests from scratch (`src/structbreaks/`) and applied them to real data. Details, figures and tables are in `notebooks/analysis.ipynb`.

| What we checked | Data | Result |
|---|---|---|
| Correctness of the implementation | Nile flow, 1871–1970 | sup-F = 75.93, break in 1898 — matches R `strucchange` |
| Critical values | Brownian bridge and DP simulation | Andrews: 8.70 (table 8.68); Bai–Perron sup-F(k), q=1: 8.43 / 7.15 / 5.92 / 4.99 / 3.85 (table 8.58 / 7.22 / 5.96 / 4.99 / 3.91) |
| Chow test at "historical" dates | food industry beta, 1960–2002 | the classical F rejects H0 at 3 of 4 dates; the HAC version is much more cautious (1987: p = 0.055) |
| Quandt–Andrews | same model | sup-F = 31.3, break in Jul 1996: beta 0.89 → 0.38 |
| CUSUM / OLS-CUSUM | same model | do **not** detect the beta shift; CUSUM² reacts to the change in variance |
| Bai–Perron (the paper's example) | US real interest rate, 1961:1–1986:3 | 2 breaks: **1972:4 and 1980:2** (Bai–Perron: 1972:3 and 1980:3 — inside our 95% CIs); regime means 1.37%, −1.65%, 5.13% |
| Sensitivity of Bai–Perron | S&P 500 volatility, 1950–2018 | number of breaks from 2 (ε = 0.25) to 8 (ε = 0.05); at ε = 0.10 BIC selects 6, LWZ 3 |
| Monte Carlo, power | T = 200, 500 replications | Chow at the true date is best; a 40-observation error in the date halves its power; sup-F lies in between; CUSUM against a beta shift ≈ 5% (no power) |
| Monte Carlo, Chow size | heteroskedasticity | with σ₂/σ₁ = 3 in the short subsample the size is 34% instead of 5%; robust Wald — 5–9% |

These results confirm the paper's main claims (Chow is more powerful when the date is known; Bai–Perron is sensitive to its settings) and quantify what the paper only states in words or omits (the weakness of CUSUM, the size distortion of the Chow test, the role of HAC).

## 7. References

1. Muthuramu P., Uma Maheswari T. (2019). Tests for Structural Breaks in Time Series Analysis: A Review of Recent Development. *Shanlax International Journal of Economics*, 7(4), 66–79.
2. Chow G. C. (1960). Tests of Equality Between Sets of Coefficients in Two Linear Regressions. *Econometrica*, 28(3), 591–605.
3. Quandt R. E. (1960). Tests of the Hypothesis that a Linear Regression System Obeys Two Separate Regimes. *JASA*, 55(290), 324–330.
4. Brown R. L., Durbin J., Evans J. M. (1975). Techniques for Testing the Constancy of Regression Relationships over Time. *JRSS B*, 37(2), 149–192.
5. Andrews D. W. K. (1993). Tests for Parameter Instability and Structural Change with Unknown Change Point. *Econometrica*, 61(4), 821–856.
6. Andrews D. W. K., Ploberger W. (1994). Optimal Tests when a Nuisance Parameter is Present Only Under the Alternative. *Econometrica*, 62(6), 1383–1414.
7. Bai J. (1997). Estimation of a Change Point in Multiple Regression Models. *Review of Economics and Statistics*, 79(4), 551–563.
8. Bai J., Perron P. (1998). Estimating and Testing Linear Models with Multiple Structural Changes. *Econometrica*, 66(1), 47–78.
9. Bai J., Perron P. (2003). Computation and Analysis of Multiple Structural Change Models. *Journal of Applied Econometrics*, 18(1), 1–22.
10. Bai J., Perron P. (2003). Critical Values for Multiple Structural Change Tests. *Econometrics Journal*, 6(1), 72–78.
11. Krämer W., Ploberger W., Alt R. (1988). Testing for Structural Change in Dynamic Models. *Econometrica*, 56(6), 1355–1369.
12. Ploberger W., Krämer W. (1992). The CUSUM Test with OLS Residuals. *Econometrica*, 60(2), 271–285.
13. Hansen B. E. (2001). The New Econometrics of Structural Change: Dating Breaks in U.S. Labor Productivity. *Journal of Economic Perspectives*, 15(4), 117–128.
14. Zeileis A., Leisch F., Hornik K., Kleiber C. (2002). strucchange: An R Package for Testing for Structural Change in Linear Regression Models. *Journal of Statistical Software*, 7(2), 1–38.
15. Inclán C., Tiao G. C. (1994). Use of Cumulative Sums of Squares for Retrospective Detection of Changes of Variance. *JASA*, 89(427), 913–923.
16. Perron P. (2006). Dealing with Structural Breaks. In: *Palgrave Handbook of Econometrics*, Vol. 1, 278–352.
17. Maddala G. S., Kim I.-M. (1998). *Unit Roots, Cointegration and Structural Change*. Cambridge University Press.
