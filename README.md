## Pricing formation equation

For a rider request with context
\(c = (h, d, w, s, a)\), where hour \(h\), day-of-week \(d\), weather \(w\),
service tier \(s\), and airport flag \(a\in\{0,1\}\), the quoted fare is:

\[
P = \Big[(\beta_0 + \beta_b) + \beta_m\,x + \beta_t\,\tau\Big]
\cdot M_{\text{tod}}(h)\cdot M_{\text{dow}}(d)\cdot M_{\text{weather}}(w)\cdot M_{\text{service}}(s)
+ a\,\beta_{\text{airport}} + f_{\text{extra}}.
\]

Where:

- \(\beta_0\): base fare,
- \(\beta_b\): booking fee,
- \(\beta_m\): per-mile coefficient,
- \(\beta_t\): per-minute coefficient,
- \(x\): trip distance in miles,
- \(\tau\): trip duration in minutes,
- \(M_{\text{tod}}\): time-of-day (surge) multiplier,
- \(M_{\text{dow}}\): day-of-week multiplier,
- \(M_{\text{weather}}\): weather multiplier,
- \(M_{\text{service}}\): service-type multiplier,
- \(\beta_{\text{airport}}\): airport additive fee,
- \(f_{\text{extra}}\): optional exogenous/additional fees.

The implemented quote is clipped at zero and rounded to 2 decimals:

\[
\widehat P = \operatorname{round}\!\left(\max(P,0),2\right).
\]

## Scenario sampling distribution

A scenario is sampled hierarchically by day context, timestep hour, and ride-specific flags.

### 1) Day and weather

\[
D \sim \operatorname{Unif}\{0,1,2,3,4,5,6\},
\]

\[
W \sim \operatorname{Categorical}(\pi^{(city)}_{\text{weather}}),
\]

with city-normalized weather probabilities:

\[
\pi^{(city)}_{k} =
\frac{\tilde\pi^{(city)}_{k}}{\sum_{j\in\mathcal W}\tilde\pi^{(city)}_{j}},
\quad k\in\mathcal W=\{\text{clear},\text{rain},\text{snow}\}.
\]

### 2) Hour-of-day

Let raw hour weights start as \(u_h=1\) for \(h\in\{0,\dots,23\}\).
Given surge windows \((s_r,e_r,m_r)\), define

\[
u_h = \prod_{r: h\in[s_r,e_r)} m_r,
\]

(with the implementation handling wrap-around windows modulo 24), then

\[
H \sim \operatorname{Categorical}(\pi_h),\qquad
\pi_h = \frac{u_h}{\sum_{j=0}^{23} u_j}.
\]

### 3) Ride-level airport and service

\[
A \sim \operatorname{Bernoulli}(p_{\text{airport}}), \quad p_{\text{airport}}=0.12,
\]

\[
S \sim \operatorname{Categorical}(\pi^{(service)}),\qquad
\pi^{(service)}=(0.85,0.15)
\text{ for }(\text{economy},\text{premium}).
\]

So the joint scenario factorization is:

\[
p(d,w,h,a,s)=p(d)\,p(w\mid city)\,p(h\mid city\;\text{surge profile})\,p(a)\,p(s).
\]

## Profile sampling distribution

A static customer pool of size \(N\) is generated once for a city and then sampled by index.

### 1) Static profile generation (per customer \(i\))

\[
\text{Age}_i \sim \operatorname{clip}\big(\mathcal N(\mu_{age},\sigma^2_{age}),18,80\big),
\]

\[
\text{Income}_i \sim \operatorname{Categorical}(\pi_{inc}),
\quad
\text{Marital}_i \sim \operatorname{Categorical}(\pi_{mar}),
\quad
\text{Gender}_i \sim \operatorname{Categorical}(\pi_{gen}),
\]

\[
\text{Household}_i \sim \operatorname{clip}(\operatorname{Poisson}(\lambda_{hh}),1,6).
\]

Employment is conditional on age:

\[
p(E=\text{Student}\mid Age<22)=0.7,
\quad p(E=\text{Employed}\mid Age<22)=0.3,
\]
\[
p(E=\text{Employed}\mid 22\le Age<65)=0.9,
\quad p(E=\text{Unemployed}\mid 22\le Age<65)=0.1,
\]
\[
p(E=\text{Retired}\mid Age\ge 65)=0.75,
\quad p(E=\text{Employed}\mid Age\ge 65)=0.25.
\]

Loyalty type and firm assignment:

\[
L_i \sim \operatorname{Bernoulli}(1-p_{new})
\quad\text{(1 = Returning, 0 = New)},
\]

\[
\text{if }L_i=1:\quad
F_i \sim \operatorname{Bernoulli}(0.5)
\text{ (Firm1 vs Firm2)},
\quad
\ell_i \sim \operatorname{Unif}(\ell_{min},\ell_{max});
\]

\[
\text{if }L_i=0:\quad F_i=\varnothing,\;\ell_i=0.
\]

All demographic parameters
\((\mu_{age},\sigma_{age},\pi_{inc},\pi_{mar},\pi_{gen},\lambda_{hh},p_{new})\)
are city-conditioned constants from priors.

### 2) Online rider sampling from pool

At interaction time, a rider is sampled uniformly from the static pool:

\[
I \sim \operatorname{Unif}\{0,1,\dots,N-1\},
\qquad
\text{Profile}=\text{Pool}[I].
\]

This yields stationary profile marginals over time while preserving persistent rider loyalty attributes.
