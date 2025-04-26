data {
  int<lower=0> N_obs;                     // Number of observed Y values
  int<lower=0> N_mis;                     // Number of missing Y values
  int<lower=1> N_total;                   // Total number of data points
  int<lower=1> K;                         // Number of predictors (incl. intercept)
  matrix[N_total, K] X;                   // Design matrix (N_total x K)
  vector[N_obs] Y_obs;                    // Observed Y values
  array[N_total] int<lower=0, upper=1> M; // Missingness indicator: 1 = observed, 0 = missing
}

parameters {
  vector<lower=0, upper=1>[N_mis] Y_mis; // Missing Y values (imputed)
  vector[K] beta;       // Regression coefficients
  real<lower=1e-3> phi; // Dispersion parameter
  real alpha0;          // Intercept for missingness model
  real alpha1;          // Slope (Y-dependent missingness)
}

model {
  // Priors
  beta ~ normal(0, 2);
  phi ~ gamma(2, 0.1);
  alpha0 ~ normal(0, 2);
  alpha1 ~ normal(0, 2);

  vector[N_total] Y;                    // Full Y vector (observed + imputed)
  vector[N_total] mu; // Mean parameter for Beta regression

  int pos_obs = 1;
  int pos_mis = 1;
  for (i in 1:N_total) {
    if (M[i] == 0) {
      Y[i] = Y_obs[pos_obs];
      pos_obs += 1;
    } else {
      Y[i] = Y_mis[pos_mis];
      pos_mis += 1;
    }
  }

  mu = inv_logit(X * beta);

  // Data model
  for (i in 1:N_total) {
    Y[i] ~ beta(mu[i] * phi, (1 - mu[i]) * phi);
  }

  // Missingness model (MNAR): depends on Y
  for (i in 1:N_total) {
    M[i] ~ bernoulli_logit(alpha0 + alpha1 * Y[i]);
  }
}

generated quantities {
  vector[N_mis] Y_mis_draw;
  // Just copy from sampled parameter (if you want draws of imputations)
  Y_mis_draw = Y_mis;
}