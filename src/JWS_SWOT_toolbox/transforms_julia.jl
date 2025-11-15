# Script for running the background transforms in Julia  

using FFTW, Interpolations, LinearAlgebra, Printf, ProgressMeter
using Base.Threads
FFTW.set_num_threads(Threads.nthreads())

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
            I += ((S[i+1] - S[i])*(sqrt(k[i]^2-k[1]^2) - sqrt(k[i+1]^2-k[1]^2)) + (S[i+1]*k[i] - S[i]*k[i+1]) * log((k[i] - sqrt(k[i]^2 - k[1]^2))/(k[i+1] - sqrt(k[i+1]^2 - k[1]^2))))/(k[i] - k[i+1])
        end
    end
    return I
end

# Forward Abel transform: 2/π \int_k^\infty Sr(κ)/\sqrt{κ^2-k^2} dκ
function abel(Sr, k)
    n = length(k)
    S1 = zeros(n)
    @printf("Forward Abel Transform\n")
    @showprogress for i = 1:n
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
    @printf("Inverse Abel Transform\n")
    @showprogress for i = 1:n
        Sr[i] = -κ[i]*sqrtint(S1p[i:n], κ[i:n])
    end
    return Sr
end

function cov(S, k)

    n = 2*(length(k)-1)
    L = 1/k[2]
    r = (0:n÷2)*L/n

    C = FFTW.r2r(S, FFTW.REDFT00)/2L # This is the TYPE-1 DCT 

    # output variance (should be unity if the variance is captured by the sampling)
    @printf("Cosine Transform\n")
    @printf("Variance from spectrum:   %8.6f\n", (S[1]/2+sum(S[2:n÷2])+S[n÷2+1]/2)/L)
    @printf("Variance from covariance: %8.6f\n", C[1])

    itp = interpolate(C, BSpline(Cubic(Line(OnGrid()))))
    itp_scaled = scale(itp, r)
    return r_values -> itp_scaled.(r_values)

end