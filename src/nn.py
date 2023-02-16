from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_utils import ModelOutput
from models import make_linear_network


class Encoder(BaseEncoder):
    def __init__(self, dims):
        BaseDecoder.__init__(self)
        self.layers = make_linear_network(
            dims,
            encoder=True
        )
        self.dims = dims

    def forward(self, x):
        out = self.layers(x)
        return ModelOutput(
            embedding=out[:, :self.dims[-1]],
            log_covariance=out[:, self.dims[-1]:]
            )
    
class Decoder(BaseDecoder):
    def __init__(self, dims):
        BaseDecoder.__init__(self)
        self.layers = make_linear_network(
            dims,
            encoder=False
        )
        self.dims = dims
        
    def forward(self, x):
        out = self.layers(x)
        return ModelOutput(
            reconstruction=out[:self.dims[-1]]
            )