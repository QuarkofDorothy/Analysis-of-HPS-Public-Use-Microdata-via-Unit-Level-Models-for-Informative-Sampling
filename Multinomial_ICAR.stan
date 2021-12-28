data {
  int<lower=1> D; // Number of counties
  int<lower=0> D_edges;
  int<lower=1, upper=D> node1[D_edges];
  int<lower=1, upper=D> node2[D_edges];
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
  real<lower=0> sigmaPhi;
  //matrix[P, K] beta; // fixed effects
  matrix[P, K-1] beta_raw;
  matrix[D, K-1] mu_raw;// random effects
  matrix[D, K-1] phi_raw;// random effects
  
}
transformed parameters{
  matrix[P, K] beta;
  matrix[D, K] mu;
  matrix[D, K] phi;
  //vector[P] zero1;
  //vector[D] zero2;
  beta = append_col(beta_raw, rep_vector(0,P));
  mu = append_col(mu_raw, rep_vector(0,D));
  phi = append_col(phi_raw, rep_vector(0,D));
}
model {
  matrix[N, K] linPred = X * beta + mu[cty,] + sigmaPhi*phi[cty,];
  vector[K] xb;
  sigmaD ~ cauchy(0, 5);
  sigmaPhi ~ cauchy(0,5);
  to_vector(mu_raw) ~ normal(0, sigmaD);  
  to_vector(beta_raw) ~ normal(0, sqrt(10));
  for(k in 1:(K-1)){
    target += -0.5 * dot_self(phi[node1,k] - phi[node2,k]);
    sum(to_vector(phi[,k])) ~ normal(0, 0.001 * D);
  }
  for(n in 1:N){
    xb = to_vector(linPred[n,]);
    target += wgt[n] * categorical_logit_lpmf(Y[n]|xb); 
  }
}
