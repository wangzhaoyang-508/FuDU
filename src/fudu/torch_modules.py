"""Optional PyTorch helpers for integrating PGUQ into detector training."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install the optional dependency with `pip install fudu-al[torch]`.") from exc


class PGUQHead(nn.Module):
    """Learnable prototype head for image-level global uncertainty."""

    def __init__(
        self,
        normal_prototypes,
        defect_prototypes,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.normal = nn.Parameter(torch.as_tensor(normal_prototypes, dtype=torch.float32))
        self.defect = nn.Parameter(torch.as_tensor(defect_prototypes, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(float(beta), dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        dn = torch.cdist(features, self.normal).min(dim=1).values
        dd = torch.cdist(features, self.defect).min(dim=1).values
        d_min = torch.minimum(dn, dd)
        ug = torch.sigmoid(self.alpha * d_min - self.beta)
        return {"ug": ug, "dn": dn, "dd": dd, "d_min": d_min}


def prototype_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    normal_prototypes: torch.Tensor,
    defect_prototypes: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """FuDU prototype contrastive loss for binary normal/defect labels."""

    dn = torch.cdist(features, normal_prototypes).min(dim=1).values
    dd = torch.cdist(features, defect_prototypes).min(dim=1).values
    labels = labels.float()
    normal_loss = dn + torch.relu(margin - dd)
    defect_loss = dd + torch.relu(margin - dn)
    return torch.where(labels < 0.5, normal_loss, defect_loss).mean()


def prototype_dispersion_loss(
    normal_prototypes: torch.Tensor,
    defect_prototypes: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """FuDU dispersion regularizer that discourages prototype collapse."""

    return _dispersion(normal_prototypes, temperature) + _dispersion(defect_prototypes, temperature)


def _dispersion(prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    if prototypes.shape[0] <= 1:
        return prototypes.new_tensor(0.0)
    distances = torch.pdist(prototypes, p=2)
    return torch.exp(-distances / float(temperature)).mean()

