using LinearAlgebra
using StatsBase
using Eirene
using RandomMatrices
using PyPlot
using Random
using Interpolations

#pyplot()
Random.seed!(0)

"Given a two-dimensional square array, replace the entries by the indices of the entries in sorted order. The results are then normalized to lie in [0, 1]"
function rankify(ar)
       n = size(ar)[1]
       mm = minimum(ar)
       ar[diagind(ar)].= mm-1
       ranked = reshape(competerank(reshape(ar, n*n)), n, n)
       ranked = ranked .- n
       ranked[diagind(ranked)].=0
       ranked = ranked ./ (n^2-n)
       return ranked
       end;

function bipartize(ar)
    ar1=convert(Array, ar)
    n = size(ar)[1]
    m = n/2
    mm = maximum(ar)
    for i in 1:n
        for j in 1:n
            if (i<=m && j<=m)
                ar1[i, j] = mm+1
            end
            if(i>m && j>m)
                ar1[i, j] = mm+1
            end
        end
    end
    return Symmetric(ar1)
end

"Converts the output of rankify into the simple Laplacian matrix (D - A) of the subgraph of the complete graph where edge values are less than p"
function graphify(ar, p)
    n = size(ar)[1]
    outar = zeros(n, n)
    for i in 1:n
        deg = 0
        for j in 1:n
            if i == j
                continue
            end
            if ar[i, j] <p
                outar[i, j] = -1
                deg+=1
            end
        end
        outar[i, i] = deg
    end
    return outar
end

"Produces the spectrum of the normalized symmetric Laplacian"
function normspec(ar)
    d = sqrt.(Diagonal(ar))
    dd = pinv(d)
    nar = dd * ar * dd
    evs = eigvals(nar)
    return evs
end

"Returns the spectral gap of the normalized Laplacian"
function specgap(ar)
    evs = normspec(ar)
    return evs[2]-evs[1]
end

"Returns the spectral gap of the non-normalized Laplacian"
function specgapraw(ar)
    evs = eigvals(ar)
    return evs[2]-evs[1]
end

"Use pyplot to plot a betti curve"
function plotbetticurve_py(D::Dict;dim=1,ocf = false, tit="")
	bcu = betticurve(D;dim = dim, ocf = ocf)
	x = bcu[:,1]
    y = bcu[:,2]
	plot(x, y, color="red", linewidth=1.0, linestyle="--")
    title(tit)
end

"Plot a list of betti curves of different complexes (but with the same dimension)"
function plotbetticurveslist_py(Dlist;dim=1,ocf = false, tit="", left=0, right=1)
    for i in Dlist
        bcu = betticurve(i;dim = dim, ocf = ocf)
        x = bcu[:,1]
        y = bcu[:,2]
        xlim(left, right)
        plot(x, y, linewidth=1.0, linestyle="--")
    end
    title(tit)
end

"Same as plotbetticurveslist_py, but with different axis scaling, for experimenting"
function plotbetticurveslist2_py(Dlist;dim=1,ocf = false, tit="", left=0, right=1)
    for i in Dlist
        bcu = betticurve(i;dim = dim, ocf = ocf)
        x = bcu[:,1]
        y = bcu[:,2]
        xscale("log")
        yscale("log")
        xlim(left, right)
        plot(x, y, linewidth=1.0, linestyle="--")
    end
    title(tit)
end

"Plot the betti curves of dimensions up to maxdim of the same filtration"
function plotbetticurvessame_py(D1::Dict;maxdim=1,ocf = false, tit="")
    for i in 0:maxdim
        bcu = betticurve(D1;dim = i, ocf = ocf)
        x = bcu[:,1]
        y = bcu[:,2]
        ll = "dimension $i"
        plot(x, y, linewidth=1.0, linestyle="--",label="dimension $i")
        
    end
    title(tit)
    legend()
end

"Given a point cloud, produce the distance function"
function distmat(z)
    ll = size(z)[2]
    dest = zeros(ll, ll)
    for i in 1:ll
        for j in 1:ll
            dest[i, j] = norm(z[:, i]-z[:, j])
        end
    end
    return dest
end

"Produce a positive rank r complex."
function posrank(r, n, dim=1)
    x = rand(r, n)
    xx = x' * x
    C = eirene(rankify(xx), maxdim=dim)
    return C
end

"Produce the rankification of a positive rank 1 random matrix v' * v, where the entries of v are uniform in [0, 1]"
function posr1(r, n)
    x = rand(r, n)
    xx = x' * x
    return rankify(xx)
end

"produce the  rankinfication of the outer product of a random gaussian rxn matrix (so sampled from the Wishart distribution) by itself."
function randr1(r, n)
    x = randn(r, n)
    xx = x' * x
    return rankify(xx)
end

"Same as above, but without the rankification"
function randrank(r, n, dim=1)
    x = randn(r, n)
    xx = x' * x
    C = eirene(rankify(xx), maxdim=dim)
    return C
end
