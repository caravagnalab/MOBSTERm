from pyro.distributions import Rejector, Pareto
from torch.distributions import constraints

class BoundedPareto(Rejector):
    def __init__(self, scale, alpha, upper_limit, validate_args=False):
        propose = Pareto(scale, alpha, validate_args=validate_args)
        self.alpha = alpha
        self.scale = scale
        self.upper_limit = upper_limit

        def log_prob_accept(x):
            """
            The result of ((x >= scale) & (x <= upper_limit)).type_as(x) is a tensor of 0s and 1s. 
            Taking the log of this tensor gives:
            - log(1) = 0 (for values x <= upper_limit or x>= lower_limit),
            - log(0) = -inf (otherwise).
            """
            # return (x <= upper_limit).type_as(x).log()
            return ((x >= scale) & (x <= upper_limit)).type_as(x).log()
        

        # log_scale = torch.Tensor(alpha) * torch.log(torch.Tensor([scale / upper_limit]))
        log_scale = 0
        super(BoundedPareto, self).__init__(propose, log_prob_accept, log_scale)
    
    @constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return constraints.interval(self.scale, self.upper_limit)
