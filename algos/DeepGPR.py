import gpytorch
import torch

import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

from torch.utils.data import TensorDataset, DataLoader








# We will use the simplest form of GP model, exact inference
class DeepGPR(DeepGP):
    def __init__(self, num_inputs, hl1_out_dims, hl2_out_dims ,train_x, train_y, batch_size):

        hidden_layer1 = ToyDeepGPHiddenLayer(
            input_dims=num_inputs,
            output_dims=hl1_out_dims,
            mean_type='linear',
        )

        hidden_layer2 = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer1.output_dims,
            output_dims=hl2_out_dims,
            mean_type='linear',
        )

        # hidden_layer3 = ToyDeepGPHiddenLayer(
        #     input_dims=hidden_layer2.output_dims,
        #     output_dims=num_hidden_dims,
        #     mean_type='linear',
        # )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer2.output_dims,
            output_dims=None,
            mean_type='constant',
        )


        super().__init__()

        
        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.last_layer = last_layer

        self.likelihood = GaussianLikelihood()
        train_dataset = TensorDataset(train_x, train_y)
        self.train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=False,)
        self.loss_data_per_epoch = list()

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer1(inputs)
        hidden_rep2 = self.hidden_layer2(hidden_rep1)
        output = self.last_layer(hidden_rep2)
        return output
    
    def optimize(self, num_epochs, num_samples, num_obs):

        optimizer = torch.optim.Adam([
            {'params': self.parameters()},
        ], lr=0.01)
        # mll = DeepApproximateMLL(PredictiveLogLikelihood(model.likelihood, model, train_x.shape[-2]))
        mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, num_obs))

        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            minibatch_iter = tqdm.tqdm(self.train_loader, desc="Minibatch", leave=False)
            for x_batch, y_batch in minibatch_iter:
                with gpytorch.settings.num_likelihood_samples(num_samples):
                    optimizer.zero_grad()
                    output = self(x_batch)
                    loss = -mll(output, y_batch)
                    loss.backward()
                    optimizer.step()

                    minibatch_iter.set_postfix(loss=loss.item())
            
            self.loss_data_per_epoch.append(loss.item())

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
    
    def infer(self, test_x, test_y):
        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=1024)

        self.eval()
        predictive_means, predictive_variances, test_lls = self.predict(test_loader)
        return predictive_means.mean(0), predictive_variances.mean(0)
        
        

## ToyDeepGPHidderLayer

class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=256, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            # gpytorch.kernels.MaternKernel(batch_shape=batch_shape,nu=1.5, ard_num_dims=input_dims),
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))