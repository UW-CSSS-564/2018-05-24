data {
  // number of observations
  int<lower = 1> N;  
  // number of covariates
  int K;
  // number of years
  int<lower = 0> J;
  // year index
  int<lower = 1, upper = J> year[N];
  // response array
  int<lower = 0, upper = 1> y[N];
  // design matrix
  matrix[N, K] X;
  // degrees of freedom
  real<lower = 0.> df_local;
  real<lower = 0.> df_global;
  real<lower = 0.> global_scale;
}
parameters {
  // intercepts
  vector[J] a_raw;
  real mu_a;
  real<lower = 0.> sigma_a;
  // regression coefficients
  vector[K] b_raw;
  // global scale
  real<lower = 0.> tau;
  // local scale
  vector<lower = 0.>[K] lambda;
}
transformed parameters {
  vector[J] a;
  vector[K] b;
  vector[N] eta;
  vector<lower = 0., upper = 1.>[N] mu;
  a = mu_a + sigma_a * a_raw;
  b = b_raw * tau .* lambda;
  for (i in 1:N) {
    eta[i] = a[year[i]] + X[i] * b;
    mu[i] = inv_logit(eta[i]);
  }
}
model {
  // priors
  a_raw ~ normal(0, 1);
  mu_a ~ normal(0., 10.);
  sigma_a ~ cauchy(0., 5.);
  b_raw ~ normal(0., 1.);
  lambda ~ student_t(df_local, 0., 1.);
  tau ~ student_t(df_global, 0., global_scale);
  // likelihood
  y ~ bernoulli(mu);
}
generated quantities {
  // simulated data
  vector[N] y_rep;
  // log-likelihood
  vector[N] log_lik;
  for (n in 1:N) {
    y_rep[n] = bernoulli_rng(mu[n]);
    log_lik[n] = bernoulli_lpmf(y[n] | mu[n]);
  }
}
