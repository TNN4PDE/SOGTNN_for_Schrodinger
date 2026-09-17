import torch
import numpy as np
import scipy
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss

def quadrature_1d(N, dtype=torch.double, device='cpu'):

    x, w = scipy.special.roots_legendre(N)  
    return (
        torch.from_numpy(x).to(dtype=dtype, device=device),
        torch.from_numpy(w).to(dtype=dtype, device=device)
    )


def composite_quadrature_1d(N, a, b, M,dtype=torch.double, device='cpu'):
    """
    Quadrature points and quadrature weights for one-dimensional Gauss-Legendre quadrature rules,
    mesh domain [a,b] into M equal subintervels and use N quadrature points in each subinterval.

    Parameters:
        N: number of quadrature points in each subintervals
        a,b: computational domain [a,b]
        M: number of subintervals of [a,b] meshed to
        dtype,device
    Returns:
        X: quadrature points size([N*M])
        W: quadrature weights size([N*M])
    """
    h = (b-a)/M
    x, w = quadrature_1d(N,dtype,device)
    x = ((x+1)/2).repeat(M)*h+torch.linspace(a,b,M+1,dtype=dtype, device=device)[:-1].repeat_interleave(N)
    w = (w/2).repeat(M)*h
    return x, w

def composite_quadrature_custom_diff_points(N_list, a, b, ratios, M_list, dtype=torch.double, device='cpu'):

    L = b - a
    ratios = np.array(ratios, dtype=np.float64)
    ratios = ratios / ratios.sum()  
    b1 = a + ratios[0] * L
    b2 = b1 + ratios[1] * L
    boundaries = [a, b1, b2, b]

    nodes_list = []
    weights_list = []

    for i in range(3):
        seg_a = boundaries[i]
        seg_b = boundaries[i + 1]
        Mi = M_list[i]
        N_current = N_list[i]
        sub_bounds = torch.linspace(seg_a, seg_b, Mi + 1, dtype=dtype, device=device)
        for j in range(Mi):
            xi = sub_bounds[j].item()
            xi1 = sub_bounds[j + 1].item()
            x_local, w_local = quadrature_1d(N_current, dtype=dtype, device=device)
            x_mapped = ((x_local + 1) / 2) * (xi1 - xi) + xi
            w_mapped = (w_local / 2) * (xi1 - xi)
            nodes_list.append(x_mapped)
            weights_list.append(w_mapped)

    x_all = torch.cat(nodes_list)
    w_all = torch.cat(weights_list)
    return x_all, w_all

def composite_quadrature_custom_diff_points_five(N_list, a, b, ratios, M_list, dtype=torch.double, device='cpu'):

    L = b - a
    ratios = np.array(ratios, dtype=np.float64)
    ratios = ratios / ratios.sum()  

    boundaries = [a]
    for r in ratios:
        boundaries.append(boundaries[-1] + r * L)

    nodes_list = []
    weights_list = []

    for i in range(5):
        seg_a = boundaries[i]
        seg_b = boundaries[i + 1]
        Mi = M_list[i]
        N_current = N_list[i]
        sub_bounds = torch.linspace(seg_a, seg_b, Mi + 1, dtype=dtype, device=device)

        for j in range(Mi):
            xi = sub_bounds[j].item()
            xi1 = sub_bounds[j + 1].item()
            x_local, w_local = quadrature_1d(N_current, dtype=dtype, device=device)

            x_mapped = ((x_local + 1) / 2) * (xi1 - xi) + xi
            w_mapped = (w_local / 2) * (xi1 - xi)
            nodes_list.append(x_mapped)
            weights_list.append(w_mapped)

    x_all = torch.cat(nodes_list)
    w_all = torch.cat(weights_list)
    return x_all, w_all



def composite_quadrature_2d(N,a1,b1,a2,b2,M1,M2,dtype=torch.double,device='cpu'):
    """
    Quadrature points and quadrature weights for two-dimensional tensor Gauss-Legendre quadrature rules,
    for [a1,b1]*[a2,b2], mesh domain [a1,b1] and [a2,b2] into M1 and M2 equal subintervels respectively,
    and use N quadrature points in one-dimensional subintervals to get tensor quadrature rules.

    Parameters:
        N: number of quadrature points in one dimension
        a1,b1,a2,b2: computational domain [a1,b1]*[a2,b2]
        M1: number of subintervals of [a1,b1] meshed to
        M2: number of subintervals of [a2,b2] meshed to
    Returns:
        X: quadrature points size([N^2*M1*M2,2])
        W: quadrature weights size([N^2*M1*M2])
    """
    p1, w1 = composite_quadrature_1d(N,a1,b1,M1,dtype=dtype,device=device)
    p2, w2 = composite_quadrature_1d(N,a2,b2,M2,dtype=dtype,device=device)
    p1 = p1.repeat(N*M2)
    p2 = p2.repeat_interleave(N*M1)
    w1 = w1.repeat(N*M2)
    w2 = w2.repeat_interleave(N*M1)
    return torch.stack((p2,p1),dim=1), w2*w1






# ******************** Unbounded Domain ********************
# Hermite-Gasuss Rule
def Hermite_Gauss_Quad(k,dtype=torch.double,device='cpu', modified=True):
    x, w = hermgauss(k)
    if modified:
        w = w*np.exp(x**2)
    return torch.from_numpy(x).to(dtype).to(device), torch.from_numpy(w).to(dtype).to(device)


# Laguerre-Gauss Rule
def Laguerre_Gauss_Quad(k,dtype=torch.double,device='cpu', modified=True):
    x, w = laggauss(k)
    if modified:
        w = w*np.exp(x)
    return torch.from_numpy(x).to(dtype).to(device), torch.from_numpy(w).to(dtype).to(device)


# 2D Tensor Laguerre-Gauss Rule
def Laguerre_Gauss_Quad2D(N,dtype=torch.double,device='cpu', modified=True):
    x, w = laggauss(N)
    if modified:
        w = w*np.exp(x)
    x = torch.from_numpy(x).to(dtype).to(device)
    w = torch.from_numpy(w).to(dtype).to(device)
    p1 = x.repeat_interleave(N)
    p2 = x.repeat(N)
    return torch.stack((p1,p2),dim=1), torch.outer(w,w).view(-1)

# Laguerre-Gauss-Radau rule
def Laguerre_Gauss_Radau(dtype=torch.double,device='cpu'):
    coord = torch.tensor([[0.000000000000000e+00, 5.882352941176471e-02],
            [2.161403052394536e-01, 2.927604493268249e-01],
            [7.263882432518047e-01, 3.181362909815314e-01],
            [1.533593160373541e+00, 2.066607692008763e-01],
            [2.644970998611911e+00, 8.994208934619415e-02],
            [4.070978160880192e+00, 2.708753007297033e-02],
            [5.825855515105604e+00, 5.679781839868921e-03],
            [7.928504185306668e+00, 8.230703112809221e-04],
            [1.040380828995104e+01, 8.100028502120838e-05],
            [1.328466107070703e+01, 5.266367668946434e-06],
            [1.661517321686662e+01, 2.173968333033556e-07],
            [2.045600602002722e+01, 5.384928271901132e-09],
            [2.489384702535191e+01, 7.374041106876865e-11],
            [3.005986292020259e+01, 4.928461815736925e-13],
            [3.617069454367918e+01, 1.309028045707531e-15],
            [4.364036518417683e+01, 9.358643168465394e-19],
            [5.352915116026845e+01, 6.770058713668848e-23]],dtype=dtype,device=device)
    return coord[:,0], coord[:,1]


def Laguerre_Gauss_Radau2D(N, dtype=torch.double,device='cpu'):
    x, w = Laguerre_Gauss_Radau(dtype=dtype,device=device)

    x = torch.from_numpy(x).to(dtype).to(device)
    w = torch.from_numpy(w).to(dtype).to(device)
    p1 = x.repeat_interleave(N)
    p2 = x.repeat(N)
    return torch.stack((p1,p2),dim=1), torch.outer(w,w).view(-1)


def low_rank_svd_approximation(k, l, p, X, Y):

    delta = 2 * k ** (2 * l)
    A = torch.exp(-((X - Y) ** 2) / delta)  # A: (N, N)

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)

    alpha = U[:, :p].T         # shape: (p, N)
    beta = Vh[:p, :]           # shape: (p, N)
    a = S[:p]                  # shape: (p,)

    sv_vectors = torch.stack([alpha, beta], dim=0)

    return sv_vectors, a


def main():
    pass


if __name__ == '__main__':
    main()
