import torch
from pcscs.extraction import extract_features_from_model


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features=torch.nn.Sequential(torch.nn.Conv2d(1,2,1),torch.nn.ReLU())
    def forward(self,x):
        return self.features(x)


def test_feature_extraction_cpu():
    model=Tiny()
    samples=[torch.zeros(1,2,2),torch.ones(1,2,2)]
    out=extract_features_from_model(samples,model,lambda x:x,device='cpu')
    assert 'features.0' in out
    assert out['features.0']['similarity_matrix'].shape==(2,2)
