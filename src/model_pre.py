import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, expression, params):
        super(CustomModel, self).__init__()
        self.params = params
        # Dynamically add parameters based on the expression
        for param in params:
            setattr(self, param, nn.Parameter(torch.randn(1)))
        self.expression = expression

    def forward(self, X):
        # Prepare the local variables for eval, including parameters and inputs
        local_vars = {"X":X, **{name: getattr(self, name) for name in self.params} }
        
        
        # Evaluate the expression safely
        try:
            psi = eval(self.expression, {"__builtins__": None}, local_vars)
        except NameError as e:
            print(f"Error in evaluating expression: {e}")
            psi = torch.tensor(0.0)  # Default behavior in case of an error

        return psi