data {
  int<lower=1> D; // Number of counties
  int<lower=0> N; // Number of observations
  int<lower=1> P; //Dimension of fixed effect coefficients
  int<lower = 2> K;
  int<lower = 1, upper = K> Y[N]; // Responses
  vector[N] wgt;  // weights
  int<lower=1, upper=D> cty[N]; // County number of each observation
  matrix[N, P] X; // Fixed effect model matrix
}
parameters {
  //vector[K] mu[D]; // random effects
  real<lower=0> sigmaD; // standard deviation of random effects
  //matrix[P, K] beta; // fixed effects
  matrix[P, K-1] beta_raw;
  matrix[D, K-1] mu_raw;// random effects
  
}
transformed parameters{
  matrix[P, K] beta;
  matrix[D, K] mu;
  //vector[P] zero1;
  //vector[D] zero2;
  beta = append_col(beta_raw, rep_vector(0,P));
  mu = append_col(mu_raw, rep_vector(0,D));
  //zero = rep_vector(0, K);
}
model {
  matrix[N, K] linPred = X * beta + mu[cty,];
  vector[K] xb;
  sigmaD ~ cauchy(0, 5);
  to_vector(mu_raw) ~ normal(0, sigmaD);  
  to_vector(beta_raw) ~ normal(0, sqrt(10));
  for(n in 1:N){
    xb = to_vector(linPred[n,]);
    target += wgt[n] * categorical_logit_lpmf(Y[n]|xb); 
  }
}
