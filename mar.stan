data {
  int<lower=0> N_obs;                               // Number of observed Y values
  int<lower=0> N_mis;                               // Number of missing Y values
  int<lower=1> N_total;                             // Total number of data points
  int<lower=1> K;                                   // Number of Predictors
  matrix[N_total, K] X;                             // Design Matrix for all observations
  vector[N_obs] Y_obs;                              // Observed values of Y
  array[N_obs] int<lower=1, upper=N_total> obs_idx; // Indices of observed Y
  array[N_mis] int<lower=1, upper=N_total> mis_idx; // Indices of missing Y
}

parameters {
  vector[2] beta;       // Beta Coefficients
  real<lower=1e-3> phi; // Dispersion Parameter for Beta Distribution
}

transformed parameters {
  vector<lower=0, upper=1>[N_obs] mu_obs;
  mu_obs = inv_logit(X[obs_idx] * beta);
}

model {
  // Priors
  beta ~ normal(0, 2);
  phi ~ gamma(2, 0.1);

  // Likelihood for observed data
  for (i in 1:N_obs) {
    Y_obs[i] ~ beta(mu_obs[i] * phi, (1 - mu_obs[i]) * phi);
  }
}

generated quantities {
  vector<lower=0, upper=1>[N_mis] Y_mis;
  vector<lower=0, upper=1>[N_mis] mu_mis;

  mu_mis = inv_logit(X[mis_idx] * beta);
  for (i in 1:N_mis)
    Y_mis[i] = beta_rng(mu_mis[i] * phi, (1 - mu_mis[i]) * phi);
}
