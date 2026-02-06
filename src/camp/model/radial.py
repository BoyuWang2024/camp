"""Radial basis functions."""
# 本模块实现了径向基函数，主要用于分子模拟中的机器学习势能函数（如MTP）。它包括径向部分的神经网络模块、切比雪夫多项式的计算以及两种包络函数。通过这些函数，可以将原子间距离和类型映射到高维特征空间，从而捕捉复杂的相互作用关系，提高模型的表达能力和预测精度。（对应论文第一部分，从原子中提取特征）
from typing import Optional

import torch
from torch import Tensor
from torch import nn as nn


class RadialPart(nn.Module):
    """Radial part of the MTP.

    f_nu_i_j(r) = \sum_\beta c_nu_i_j * radial_basis_\beta(r)

    Eq. 3 of Shapeev.
    """

    def __init__(# 初始化
        self,
        n_u: int,# 径向基函数的数量(通道数：类比卷积神经网络中的卷积核数量。 每个基函数可以看作是一个卷积核，用于提取输入数据中的特定特征，多个通道可以捕捉到更多样化的特征，从而提升模型的表达能力和性能。)
        n_z: int,# 原子类型的数量(不同种类的原子数目)（为不同类型原子对分别训练一套系数，提高模型准确性）
        max_chebyshev_degree: int = 9,# 切比雪夫多项式的最高次数(基函数中切比雪夫多项式的最高次数，决定了基函数的复杂度和表达能力，更高的次数可以捕捉到更复杂的距离依赖关系，但也可能增加计算复杂度和过拟合风险)
        r_cut: float = 5,# 截断距离(在分子模拟中，r_cut表示原子间相互作用的截断距离，超过该距离的相互作用被忽略，以提高计算效率。选择合适的r_cut值对于准确描述系统的物理性质至关重要)
        envelope: Optional[int] = None,# 包络函数(包络函数用于调整径向基函数在截断距离处的行为，使其平滑地过渡到零，从而避免由于截断引入的不连续性和数值不稳定性。如果envelope为None，则使用MTP中的二次多项式包络函数；否则，envelope是一个正整数p，表示使用Dimenet中的包络函数，该函数的形式取决于p的值)
    ):
        """
        Args:
            n_u: number of radial basis functions.
            n_z: number of atom types.
            max_chebyshev_degree: max degree of the Chebyshev polynomial. The total
                number of chebyshev polynomials is `max_chebyshev_degree + 1`; +1 for
                the zeroth degree.
            r_cut: cutoff distance.
            envelope: envelope function to make the radial basis function smooth at
                r_cut. if None, using the MTP 2nd order polynomial envelope. Otherwise,
                p is a positive integer, and the envelope function in dimenet is used.
        """
        super().__init__()

        self.n_u = n_u
        self.n_z = n_z
        self.max_chebyshev_degree = max_chebyshev_degree
        self.r_cut = r_cut
        self.envelope = envelope

        self.c = nn.Parameter(torch.empty(n_z, n_z, n_u, max_chebyshev_degree + 1))
        self.reset_parameters()

    def reset_parameters(self):# 将权重初始化为均匀分布（初始化函数）
        """Initialize the weights to:

            uniform(-1/sqrt(in_features), 1/sqrt(in_features)).

        Note, self.c can be regarded as a collection of multiple linear layers, each for
        a specific combination of zi and zj.

        https://github.com/pytorch/pytorch/blob/e3ca7346ce37d756903c06e69850bdff135b6009/torch/nn/modules/linear.py#L109
        """
        k = 1 / (self.max_chebyshev_degree + 1) ** 0.5
        nn.init.uniform_(self.c, -k, k)

    def forward(self, r: Tensor, zi: Tensor, zj: Tensor):# 径向部分的前向传播(将径向距离（乘以包络函数的切尔雪夫多项式）和原子类型映射到径向特征（通过乘以可学习的系数）)
        """
        Args:
            r: 1D tensor of distances between atoms i and j.
            zi: 1D tensor of integers. type of atom i. The choice are 0, 1, 2, ...
                the number of atom types.
            zj: 1D tensor of integers. type of atom j. The choice are 0, 1, 2, ...
                the number of atom types.

        Note:
            The shape of r, zi, and zj should be the same.

        Returns:
            A tensor of shape (len(r), n_nu). The first dimension corresponds to `nu`
            in Eq. 3 of Shapeev, and the second dimension denotes the size of the
            distances.
        """
        # shape (n_nu, len(r))
        radial = radial_basis(
            self.max_chebyshev_degree, r, r_cut=self.r_cut, envelope=self.envelope
        )

        # select c for r according to zi and zj
        c = self.c[zi, zj, :, :]  # shape(len(r), n_nu, len(degrees))

        # linear combination of radial basis functions of different degrees
        out = torch.einsum("rub, br -> ru", c, radial)

        return out


def radial_basis(# 径向基函数，使用切比雪夫多项式（切尔雪夫多项式乘包络函数）
    degree: int,
    r: Tensor,
    r_min: float = 0,
    r_cut: float = 5,
    envelope: Optional[int] = None,
) -> Tensor:
    """
    Radial basis function, using Chebyshev polynomials.

    I.e. Q in Eq. 4 of Shapeev.

    Args:
        degree: max degree of the Chebyshev polynomial to use.
        r: distance, 1D tensor.
        r_min: minimum distance.
        r_cut: cutoff distance.
        envelope: envelope function to make the radial basis function smooth at r_cut.
            if None, using the MTP 2nd order polynomial envelope. Otherwise, p is a
            positive integer, and the envelope function in dimenet is used.

    Returns:
        A tensor X of shape (degree+1, *r.shape); +1 to include the zeroth degree.
        The first dimension denotes the degree of the polynomial. X[i] is the result
        for the i-th degree polynomial.
    """
    # select r < r_cut ones for computation
    mask = r < r_cut
    selected_r = r[mask]

    # normalize r to [0, 1]
    normalized_r = (selected_r - r_min) / (r_cut - r_min)

    che = chebyshev_first(degree, normalized_r)

    if envelope is None:
        env = mtp_envelope(normalized_r)
    else:
        env = dimenet_envelope(normalized_r, p=envelope)

    Q = che * env

    # prepare output
    shape = torch.Size([degree + 1]) + r.shape
    out = torch.zeros(shape, dtype=r.dtype, device=r.device)
    out[:, mask] = Q

    return out


def chebyshev_first(n: int, x: Tensor) -> Tensor:# 这是切比雪夫多项式（用于拟合径向特征的基函数的多项式）
    """Chebyshev polynomials of the first kind.

    Args:
        n: highest degree of the polynomial to compute.
        x: input tensor.

    Returns:
        A tensor of shape (n + 1, *x.shape). The first dimension denotes the degree
        of the polynomial, e.g. T[1] is the result of the first degree polynomial.
    """
    T = [torch.ones_like(x), x]  # T0 and T1
    for i in range(2, n + 1):
        T.append(2.0 * x * T[i - 1] - T[i - 2])

    T = torch.stack(T, dim=0)

    return T

# 两种包络函数
def mtp_envelope(r: Tensor):# 这是两种包络函数之一：MTP 包络
    """The envelope function used in the MTP."""
    return (1 - r) ** 2


def dimenet_envelope(r: Tensor, p: int = 6):# 这是两种包络函数之一：DimNet 包络
    """The envelope function used in DimNet.

    1 - (p+1)(p+2)/2*x**p + p*(p+2)*x**(p+1) - p*(p+1)/2*x**(p+2)

    This is also the envelope function used hybrid NN of Mingjian Wen when p = 3.
    """
    if p == 6:
        return 1 - 28 * r**6 + 48 * r**7 - 21 * r**8
    elif p == 3:
        return 1 - 10 * r**3 + 15 * r**4 - 6 * r**5
    else:
        return (
            1
            - (p + 1) * (p + 2) / 2 * r**p
            + p * (p + 2) * r ** (p + 1)
            - p * (p + 1) / 2 * r ** (p + 2)
        )
