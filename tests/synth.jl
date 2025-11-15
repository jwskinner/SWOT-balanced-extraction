using FFTW, Interpolations, LinearAlgebra, PyPlot, Printf, ProgressMeter, Base.Threads

function cov(S, k)

    n = 2*(length(k)-1)
    L = 1/k[2]
    r = (0:n÷2)*L/n

    C = FFTW.r2r(S, FFTW.REDFT00)/2L

    # output variance (should be unity if the variance is captured by the sampling)
    @printf("Variance from spectrum:   %8.6f\n", (S[1]/2+sum(S[2:n÷2])+S[n÷2+1]/2)/L)
    @printf("Variance from covariance: %8.6f\n", C[1])

    itp = interpolate(C, BSpline(Cubic(Line(OnGrid()))))
    return scale(itp, r)

end

# Calculate \int_k^\infty S(κ)/\sqrt{κ^2-k^2} dk
function sqrtint(S, k)
    I = 0.0
    if k[1] == 0.0
        I += S[2]
        for i = 2:length(S)-1
            I += S[i+1] - S[i] + (S[i+1]*k[i] - S[i]*k[i+1]) * log(k[i+1]/k[i])/(k[i] - k[i+1])
        end
    else
        for i = 1:length(S)-1
            I += ((S[i+1] - S[i])*(sqrt(k[i]^2-k[1]^2) - sqrt(k[i+1]^2-k[1]^2)) + (S[i+1]*k[i] - S[i]*k[i+1]) * log((k[i] - sqrt(k[i]^2-k[1]^2))/(k[i+1] - sqrt(k[i+1]^2-k[1]^2))))/(k[i] - k[i+1])
        end
    end
    return I
end

# Forward Abel transform: 2/π \int_k^\infty Sr(κ)/\sqrt{κ^2-k^2} dκ
function abel(Sr, k)
    n = length(k)
    S1 = zeros(n)
    @showprogress @threads for i = 1:n
        S1[i] = 2/π*sqrtint(Sr[i:n], k[i:n])
    end
    return S1
end

# Inverse Abel transform: -κ \int_κ^\infty S1'(k)/\sqrt{k^2-κ^2} dk
function iabel(S1, κ)
    n = length(κ)
    dκ = κ[2] - κ[1]
    S1p = [-3S1[1] + 4S1[2] - S1[3]; S1[3:n] - S1[1:n-2]; S1[n-2] - 4S1[n-1] + 3S1[n]]/2dκ
    Sr = zeros(n)
    @showprogress @threads for i = 1:n
        Sr[i] = -κ[i]*sqrtint(S1p[i:n], κ[i:n])
    end
    return Sr
end

nx = 200
ny = 60
ng = 10
Δk = 2e3

nn = 59
Δn = 6.8e3

Ab = 2.7e3
λb = 223.4e3
sb = 4.7

An = 0.0 # 4.36
λn = 100e3
sn = 1.7

σ = 0.0 # 5.2e-2

g = 9.81
f = 1e-4

xk = (0.5:nx-0.5)*Δk
yk = [-ny÷2+0.5:-ng÷2-0.5; ng÷2+0.5:ny÷2-0.5]*Δk

xn = (0.5:nn-0.5)*Δn
yn = zeros(nn)

xs = (0.5:nx-0.5)*Δk
ys = (-ny÷2+0.5:ny÷2-0.5)*Δk

ρ = 2π*4e3
δ = π*Δk/2log(2)

B(k) = Ab/(1 + (λb*k)^sb)
N(k) = An/(1 + (λn*k)^2)^(sn/2)

k = (0:200_000)/1e7
ckk = cov(abel(iabel(B.(k)+N.(k), k).*exp.(-δ^2*k.^2), k), k)
cnn = cov(B.(k), k)
ckn = cov(abel(iabel(B.(k), k).*exp.(-δ^2*k.^2/2), k), k)
css = cov(abel(iabel(B.(k), k).*exp.(-ρ^2*k.^2), k), k)
csk = cov(abel(iabel(B.(k), k).*exp.(-(ρ^2+δ^2)*k.^2/2), k), k)
csn = cov(B.(k), k)

Xk = reshape(xk.*ones(ny-ng)', nx*(ny-ng))
Yk = reshape(ones(nx).*yk', nx*(ny-ng))

Xs = reshape(xs.*ones(ny)', nx*ny)
Ys = reshape(ones(nx).*ys', nx*ny)

rkk = hypot.(Xk .- Xk', Yk .- Yk')
rnn = hypot.(xn .- xn', yn .- yn')
rkn = hypot.(Xk .- xn', Yk .- yn')
rss = hypot.(Xs .- Xs', Ys .- Ys')
rsk = hypot.(Xs .- Xk', Ys .- Yk')
rsn = hypot.(Xs .- xn', Ys .- yn')

Rkk = ckk.(rkk)
Rnn = ckk.(rnn) + σ^2*I
Rkn = ckk.(rkn)
Rss = css.(rss)
Rsk = csk.(rsk)
Rsn = csk.(rsn)

# draw from prior
F = cholesky(Rss + 1e-15I)
h = F.L*randn(nx*ny)

# calculate posterior covariance
F = cholesky([Rkk Rkn; Rkn' Rnn] + 1e-15I)
v = F.L\[Rsk Rsn]'
C = Rss - v'*v

# draw from posterior
F = cholesky(C + 1e-15I)
η = F.L*randn(nx*ny)

h = reshape(h, (nx, ny))
η = reshape(η, (nx, ny))

ζ = g/f*(h[1:nx-2,2:ny-1] + h[3:nx,2:ny-1] + h[2:nx-1,1:ny-2] + h[2:nx-1,3:ny] - 4h[2:nx-1,2:ny-1])/Δk^2
ε = g/f*(η[1:nx-2,2:ny-1] + η[3:nx,2:ny-1] + η[2:nx-1,1:ny-2] + η[2:nx-1,3:ny] - 4η[2:nx-1,2:ny-1])/Δk^2

fig, ax = subplots(4, 1; sharex=true, sharey=true, figsize=(6.4, 9.6))
vmax = maximum(abs.(h))
img1 = ax[1].imshow(h'; extent=1e-3Δk*[0, nx, -ny÷2, ny÷2], cmap="RdBu_r", vmin=-vmax, vmax)
vmax = maximum(abs.(ζ))/f
img2 = ax[2].imshow(ζ'/f; extent=1e-3Δk*[1, nx-1, -ny÷2+1, ny÷2-1], cmap="RdBu_r", vmin=-vmax, vmax)
vmax = maximum(abs.(η))
img3 = ax[3].imshow(η'; extent=1e-3Δk*[0, nx, -ny÷2, ny÷2], cmap="RdBu_r", vmin=-vmax, vmax)
vmax = maximum(abs.(ε))/f
img4 = ax[4].imshow(ε'/f; extent=1e-3Δk*[1, nx-1, -ny÷2+1, ny÷2-1], cmap="RdBu_r", vmin=-vmax, vmax)
ax[1].set_title("prior draw height (m)")
ax[2].set_title(L"prior draw vorticity ($f$)")
ax[3].set_title("posterior draw height (m)")
ax[4].set_title(L"posterior draw vorticity ($f$)")
ax[1].set_xlim(0, 1e-3Δk*nx)
ax[1].set_ylim(-1e-3Δk*ny÷2, 1e-3Δk*ny÷2)
fig.tight_layout()
colorbar(img1; ax=ax[1], fraction=0.06, pad=0.025, shrink=0.75)
colorbar(img2; ax=ax[2], fraction=0.06, pad=0.025, shrink=0.75)
colorbar(img3; ax=ax[3], fraction=0.06, pad=0.025, shrink=0.75)
colorbar(img4; ax=ax[4], fraction=0.06, pad=0.025, shrink=0.75)
