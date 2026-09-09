import gpytorch
import torch

# We will use the simplest form of GP model, exact inference
class GPR(gpytorch.models.ExactGP):
    def __init__(self, train_X, train_y, likelihood, kernel):
        super(GPR, self).__init__(train_X, train_y, likelihood)
        self.likelihood = likelihood
        self.train_X = train_X
        self.train_y = train_y
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def optimize(self, training_iter):
        # Find optimal model hyperparameters
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(self.train_X)
            # Calc loss and backprop gradients
            loss = -mll(output, self.train_y)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item()
            # ))
            # if not (i % 100):
            #     print(i)
            optimizer.step()
    
    def infer(self, X):
        # Get into evaluation (predictive posterior) mode
        self.eval()
        self.likelihood.eval()

        # Test points are regularly spaced along [0,1]
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self(X))

        return observed_pred
        
