using Random, Distributions, StatsBase, LinearAlgebra, DelimitedFiles, Optim, LineSearches, Flux
using BSON: @save  ## this is to store trained flux model to be loaded for future usage
Random.seed!()

# This function computes the approximate ELBO of the model
function ELBO(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, B::Array{Float64, 2}, ρ::Float64, μ, Y::Array{Float64, 2}, K::Int, N::Int)
    apprELBO = 0.0
    inv_Σ = inv(Σ)
    apprELBO = 0.5 * N * log(det(inv_Σ))
    for i in 1:N
        apprELBO -= 0.5 * (dot(λ[:,i] .- μ[:,i], inv_Σ, λ[:,i] .- μ[:,i]) + tr(inv_Σ*ν[:,:,i])) #first 2 terms
    end
    if isnan(apprELBO)
        println("ERROR 1")
        #break
    end

    #=for k in 1:K
        for j in 1:N
            for i in 1:N
                if i != j
                    apprELBO -= ϕ[i,j,k]*log(ϕ[i,j,k]) #last entropic term
                end
            end
        end
    end=#
    if isnan(apprELBO)
        println("ERROR 2")
        #break
    end

    for j in 1:N
        for i in 1:N
            if i != j
                apprELBO += dot(ϕ[i,j,:],λ[:,i])# vcat(λ[:,i], 1.0)) #third term
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 3")
        #break
    end

    for i in 1:N
        theta = zeros(K)
        #theta .= exp.(λ[:,i])  #exp.(vcat(λ[:,i], 1.0))
        #theta /= sum(theta)
        theta .= softmax(λ[:,i])
        # second line of the expression above
        apprELBO -= (N-1) * ( (log(sum(exp.(λ[:,i])))) + 0.5*tr((diagm(theta) .- theta * theta')*ν[:,:,i]) )
        #gaussian entropic term
        apprELBO += 0.5*log(det(ν[:,:,i]))
    end
    if isnan(apprELBO)
        println("ERROR 4")
        #break
    end

    #likelihood term
    logB = log.(B*(1-ρ))
    logB2 = log.(ones(K,K) .- (1-ρ)*B)
    for i in 1:N
        for j in 1:i-1
            for k in 1:K
                for g in 1:K
                    #logP = Y[i,j] * ( log(B[k,g]*(1-ρ)) - abs(i-j)/N) - B[k,g]*(1-ρ)*exp(-abs(i-j)/N)
                    #logP = Y[i,j]*log(B[k,g]*(1-ρ)) + (1-Y[i,j])*log(1-B[k,g]*(1-ρ))
                    logP = Y[i,j]*logB[k,g] + (1-Y[i,j])*logB2[k,g]
                    apprELBO += ϕ[i,j,k]*ϕ[j,i,g]*logP
                end
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 5")
        #break
    end
    return apprELBO
end

################################################################################

# This function computes the approximate ELBO of the model with the gaussian likelihood
function ELBO_gauss(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, B::Array{Float64, 2}, μ, Y::Array{Float64, 2}, K::Int, N::Int, like_var::Array{Float64, 2})
    apprELBO = 0.0
    inv_Σ = inv(Σ)
    #inv_Σ = Matrix(1.0I, length(Σ[1,:]), length(Σ[1,:]))
    apprELBO = 0.5 * N * log(det(inv_Σ))
    for i in 1:N
        apprELBO -= 0.5 * (dot(λ[:,i] .- μ[:,i], inv_Σ, λ[:,i] .- μ[:,i]) + tr(inv_Σ*ν[:,:,i])) #first 2 terms
    end

    if isnan(apprELBO)
        println("ERROR 1")
        #break
    end

    for k in 1:K
        for j in 1:N
            for i in 1:N
                if i != j && ϕ[i,j,k] > eps()
                    apprELBO -= ϕ[i,j,k]*log(ϕ[i,j,k]) #last entropic term
                end
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 2")
        #break
    end

    for j in 1:N
        for i in 1:N
            if i != j
                apprELBO += dot(ϕ[i,j,:],λ[:,i])# vcat(λ[:,i], 1.0)) #third term
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 3")
        #break
    end

    for i in 1:N
        theta = zeros(K)
        #theta .= exp.(λ[:,i])  #exp.(vcat(λ[:,i], 1.0))
        #theta /= sum(theta)
        theta .= softmax(λ[:,i])
        # second line of the expression above
        apprELBO -= (N-1) * ( (log(sum(exp.(λ[:,i])))) + 0.5*tr((diagm(theta) .- theta * theta')*ν[:,:,i]) )
        #gaussian entropic term
        apprELBO += 0.5*log(det(ν[:,:,i]))
    end
    if isnan(apprELBO)
        println("ERROR 4")
        #break
    end
    #println("Partial ELBO: ", apprELBO)
    #likelihood term
    for i in 1:N
        for j in 1:i-1
            for k in 1:K
                for g in 1:K
                    #logP = -(0.5*(Y[i,j] - B[k,g])^2)/like_var[1] -0.5*log(like_var[1])
                    logP = -(0.5*(Y[i,j] - B[k,g])^2)/like_var[k,g] -0.5*log(like_var[k,g])
                    apprELBO += ϕ[i,j,k]*ϕ[j,i,g]*logP
                end
            end
        end
    end
    #apprELBO += -0.25*N*(N-1)*log(like_var[1])
    if isnan(apprELBO)
        println("ERROR 5")
        #break
    end
    return apprELBO
end



# This function computes the approximate ELBO of the model with the gaussian likelihood
# but with a shared mean vector for the logistic normal prior instead of a node covariate
# dependent one.
function ELBO_gauss(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2},B::Array{Float64, 2}, μ, Y::Array{Float64, 2}, K::Int, N::Int, like_var::Array{Float64, 2})
    apprELBO = 0.0
    inv_Σ = inv(Σ)
    apprELBO = 0.5 * N * log(det(inv_Σ))
    for i in 1:N
        apprELBO -= 0.5 * (dot(λ[:,i] .- μ, inv_Σ, λ[:,i] .- μ) + tr(inv_Σ*ν[:,:,i])) #first 2 terms
    end

    if isnan(apprELBO)
        println("ERROR 1")
        #break
    end

    for k in 1:K
        for j in 1:N
            for i in 1:N
                if i != j && ϕ[i,j,k] > eps()
                    apprELBO -= ϕ[i,j,k]*log(ϕ[i,j,k]) #last entropic term
                end
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 2")
        #break
    end

    for j in 1:N
        for i in 1:N
            if i != j
                apprELBO += dot(ϕ[i,j,:],λ[:,i])# vcat(λ[:,i], 1.0)) #third term
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 3")
        #break
    end

    for i in 1:N
        theta = zeros(K)
        #theta .= exp.(λ[:,i])  #exp.(vcat(λ[:,i], 1.0))
        #theta /= sum(theta)
        theta .= softmax(λ[:,i])
        # second line of the expression above
        apprELBO -= (N-1) * ( (log(sum(exp.(λ[:,i])))) + 0.5*tr((diagm(theta) .- theta * theta')*ν[:,:,i]) )
        #gaussian entropic term
        apprELBO += 0.5*log(det(ν[:,:,i]))
    end
    if isnan(apprELBO)
        println("ERROR 4")
        #break
    end
    #println("Partial ELBO: ", apprELBO)
    #likelihood term
    for i in 1:N
        for j in 1:i-1
            for k in 1:K
                for g in 1:K
                    #logP = -(0.5*(Y[i,j] - B[k,g])^2)/like_var[1] -0.5*log(like_var[1])
                    logP = -(0.5*(Y[i,j] - B[k,g])^2)/like_var[k,g] -0.5*log(like_var[k,g])
                    apprELBO += ϕ[i,j,k]*ϕ[j,i,g]*logP
                end
            end
        end
    end
    #apprELBO += -0.25*N*(N-1)*log(like_var[1])
    if isnan(apprELBO)
        println("ERROR 5")
        #break
    end
    return apprELBO
end



#=
In the E-step due to the non-conjugancy of the logistic normal with the multinomial
we resort to a gaussian approximation of the variational porsterior (Wang, Blei, 2013).
The approximations equals an optimizetion process that we characterize with the followfin functions
=#

function f(η_i::Array{Float64, 1}, ϕ_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i, N::Int)
    #f = 0.5 * dot(η_i .- μ_i, inv_Σ, η_i .- μ_i) - dot(η_i, ϕ_i) +(N-1)*log(sum(exp.(η_i)))
    f = 0.5 * dot(η_i .- μ_i, inv_Σ, η_i .- μ_i) - dot(η_i, ϕ_i) +(N-1)*log(sum(exp.(η_i)))

    #MMM = Matrix(1.0I, length(inv_Σ[1,:]), length(inv_Σ[1,:]))
    #f = 0.1 * 0.5 * dot(η_i .- μ_i, η_i .- μ_i) - dot(η_i, ϕ_i) +(N-1)*log(sum(exp.(η_i)))

    #f = 0.5 * dot(η_i .- μ_i, inv_Σ, η_i .- μ_i) - dot(η_i, ϕ_i)/(N-1) +log(sum(exp.(η_i)))
    return f
end

function gradf!(G, η_i::Array{Float64, 1}, ϕ_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i, N::Int)
    G .= softmax(η_i)*(N-1) .- ϕ_i .+ inv_Σ*(η_i .- μ_i)

    #MMM = Matrix(1.0I, length(inv_Σ[1,:]), length(inv_Σ[1,:]))
    #G .= softmax(η_i)*(N-1) .- ϕ_i .+ 0.1 * (η_i .- μ_i)
    #G .= exp.(η_i)/sum(exp.(η_i))*(N-1) .- ϕ_i .+ inv_Σ*(η_i .- μ_i)
    #G .= exp.(η_i)/sum(exp.(η_i)) .- ϕ_i/(N-1) .+ inv_Σ*(η_i .- μ_i)
end

function hessf!(H, η_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i, N::Int)
    #theta = exp.(η_i)/sum(exp.(η_i))
    theta = softmax(η_i)
    H .=  (N-1)*(diagm(theta) .- theta*theta') .+ inv_Σ

    #MMM = Matrix(1.0I, length(inv_Σ[1,:]), length(inv_Σ[1,:]))
    #H .=  (N-1)*(diagm(theta) .- theta*theta') .+ 0.1 * MMM
    #H .=  (diagm(theta) .- theta*theta') .+ inv_Σ
end


################################################################################
# Function that perform the variational optimization
function Estep_logitNorm!(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    inv_Σ::Array{Float64, 2}, μ, N::Int, K::Int)
    G = zeros(K)
    H = zeros(K,K)
    for i in 1:N
        ϕ_i = sum(ϕ[i,:,:],dims=1)[1,:]
        μ_i = μ[:,i]
        res = optimize(η_i -> f(η_i, ϕ_i, inv_Σ, μ_i, N), (G, η_i) -> gradf!(G,η_i, ϕ_i, inv_Σ, μ_i, N), randn(K), BFGS(linesearch = LineSearches.BackTracking(order=2)))#BFGS())
        η_i = Optim.minimizer(res)
        hessf!(H, η_i, inv_Σ, μ_i, N)
        λ[:,i] .= η_i
        ν[:,:,i] .= Hermitian(inv(H))
    end
end
# without covariates
function Estep_logitNorm_noX!(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    inv_Σ::Array{Float64, 2}, μ, N::Int, K::Int)
    G = zeros(K)
    H = zeros(K,K)
    for i in 1:N
        ϕ_i = sum(ϕ[i,:,:],dims=1)[1,:]
        res = optimize(η_i -> f(η_i, ϕ_i, inv_Σ, μ, N), (G, η_i) -> gradf!(G,η_i, ϕ_i, inv_Σ, μ, N), randn(K), BFGS(linesearch = LineSearches.BackTracking(order=2)))#BFGS())
        η_i = Optim.minimizer(res)
        hessf!(H, η_i, inv_Σ, μ, N)


        ###
        #η_i .-= η_i[K]

        ###


        λ[:,i] .= η_i
        ν[:,:,i] .= Hermitian(inv(H))
    end
end

function Estep_multinomial!(ϕ::Array{Float64, 3}, λ, B::Array{Float64, 2},
    ρ::Float64, Y::Array{Float64, 2}, N::Int, K::Int)
    for i in 1:N
        for j in 1:N
            if i != j
                for k in 1:K
                    logPi = λ[k,i]
                    for g in 1:K
                        logPi += ϕ[j,i,g] * (Y[i,j]*log(B[k,g]*(1-ρ)) + (1-Y[i,j])*log(1-(1-ρ)*B[k,g]))
                    end
                    #ϕ[i,j,k] = exp(logPi)
                    ϕ[i,j,k] = logPi
                end
                #ϕ[i,j,:] ./= sum(ϕ[i,j,:])
                ϕ[i,j,:] = softmax(ϕ[i,j,:])
            end
        end
    end
end

function Estep_multinomial_gauss!(ϕ::Array{Float64, 3}, λ, B::Array{Float64, 2},
     Y::Array{Float64, 2}, N::Int, K::Int, like_var::Array{Float64, 2})
    for i in 1:N
        for j in 1:N
            if i != j
                for k in 1:K
                    logPi = λ[k,i]
                    for g in 1:K
                        #logPi += -0.5 * ϕ[j,i,g] * (Y[i,j]*log(B[k,g]*(1-ρ)) + (1-Y[i,j])*log(1-(1-ρ)*B[k,g]))
                        #logPi += -ϕ[j,i,g] * ((Y[i,j] - B[k,g])^2) / (2*like_var[1])
                        logPi += -ϕ[j,i,g] *( ((Y[i,j] - B[k,g])^2) / (2*like_var[k,g])  + 0.5*log(like_var[k,g]))
                    end
                    #ϕ[i,j,k] = exp(logPi)
                    ϕ[i,j,k] = logPi
                end
                #ϕ[i,j,:] ./= sum(ϕ[i,j,:])
                ϕ[i,j,:] = softmax(ϕ[i,j,:])
            end
        end
    end
end


################################################################################
################################################################################
################################################################################

#function Mstep_logitNorm!(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
#    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, μ, Γ::Array{Float64, 2}, X::Array{Float64, 2}, N::Int, K::Int, P::Int)
#    for m in 1:20
#        for k in 1:K
#            #Γ[k,:] = inv(X*X' + diagm(ones(P)/σ_2[k]))*(X*λ[k,:])
#            #σ_2[k] = (0.5 + sum(Γ[k,:].^2))/(0.5 + P)
#            Γ[k,:] = inv(X*X' + diagm(ones(P)/σ_2[k]))*(X*λ[k,:])
#            σ_2[k] = (2 + sum(Γ[k,:].^2))/(2 + P)
#        end
#
#    end
#
#    μ = Γ * X
#
#    Σ .= zeros(K,K)
#    for i in 1:N
#        Σ .+= 1/N * (ν[:,:,i] .+ (λ[:,i] .- μ[:,i])*(λ[:,i] .- μ[:,i])')
#    end
#end

function Mstep_logitNorm!(λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, μ, Γ::Array{Float64, 2}, X::Array{Float64, 2}, N::Int, K::Int, P::Int)
    a0 = 1.0
    b0 = 1.0

    for k in 1:K
        β_N = 0.5
        for m in 1:20
            σ_2[k] = (b0 + 0.5*sum(Γ[k,:].^2)) / (a0 + 0.5*P)
            S_N = inv(β_N*X*X' + diagm(ones(P)/σ_2[k]))
            Γ[k,:] = β_N * S_N*(X*λ[k,:])
            β_N = N/2 * (dot(λ[k,:],λ[k,:])/2 - dot(Γ[k,:], X*λ[k,:]) + 0.5*tr((X * X')*(Γ[k,:]*Γ[k,:]' .+ S_N)) )^(-1)
        end
        #println(β_N)
    end

    μ = Γ * X

    Σ .= zeros(K,K)
    for i in 1:N
        Σ .+= 1/N * (ν[:,:,i] .+ (λ[:,i] .- μ[:,i])*(λ[:,i] .- μ[:,i])')
    end
    Σ .= Hermitian(Σ)
end



function Mstep_logitNorm_flux!(λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, μ, Γ::Array{Float64, 2}, X::Array{Float64, 2}, N::Int, K::Int, P::Int)

    ####
    lr = 0.01
    model = Chain(Dense(P,K, bias=false))
    model[1].weight .= Float32.(Γ)
    ps = Flux.params(model)
    opt = ADAM(lr)
    L(a,b) = (Flux.Losses.kldivergence(softmax(model(a)), softmax(b)))
    max_iter = 50
    for i in 1:max_iter
        gs = gradient(() -> L(Float32.(X), Float32.(λ)), ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    Γ .= model[1].weight
    μ = Γ * X
    ####

    Σ .= zeros(K,K)
    for i in 1:N
        Σ .+= 1/N * (ν[:,:,i] .+ (λ[:,i] .- μ[:,i])*(λ[:,i] .- μ[:,i])')
    end
    Σ .= Hermitian(Σ)
end

function Mstep_logitNorm_flux_multi!(λ, ν::Vector{Array{Float64, 3}},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, μ, Γ::Array{Float64, 2}, X::Array{Float64, 2}, N::Int, K::Int, P::Int, n_regions::Int)

    ####
    lr = 0.01
    model = Chain(Dense(P,K, bias=false))
    model[1].weight .= Float32.(Γ)
    ps = Flux.params(model)
    opt = ADAM(lr)
    L(a,b) = (Flux.Losses.kldivergence(softmax(model(a)), softmax(b)))
    max_iter = 50
    for i in 1:max_iter
        gs = gradient(() -> L(Float32.(X), Float32.(λ)), ps)
        Flux.Optimise.update!(opt, ps, gs)
    end
    Γ .= model[1].weight
    μ = Γ * X
    ####

    Σ .= zeros(K,K)
    for i_region in 1:n_regions
        for i in 1:N
            Σ .+= 1/(N*n_regions) * (ν[i_region][:,:,i] .+ (λ[:,i+(i_region-1)*N] .- μ[:,i+(i_region-1)*N])*(λ[:,i+(i_region-1)*N] .- μ[:,i+(i_region-1)*N])')
        end
    end
    Σ .= Hermitian(Σ)
end



function Mstep_logitNorm_noX!(λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, μ, N::Int, K::Int)
    μ .= zeros(K)


    for i in 1:N
        μ .+= 1/N * λ[:,i]
        #μ .= inv(Matrix(1.0I,K,K) .+ Σ/N)*μ
    end
    Σ .= zeros(K,K)
    for i in 1:N
        Σ .+= 1/N * (ν[:,:,i] .+ (λ[:,i] .- μ)*(λ[:,i] .- μ)')
    end
    Σ .= Hermitian(Σ)
end

################################################################################


function Mstep_blockmodel!(ϕ::Array{Float64, 3}, B::Array{Float64, 2}, ρ::Float64,
    Y::Array{Float64, 2}, N::Int, K::Int)
    for k in 1:K
        for g in 1:K
            num = 0.
            den = 0.
            for j in 1:N
                for i in 1:N
                    phi_prod = ϕ[i,j,k]*ϕ[j,i,g]
                    num += phi_prod*Y[i,j]
                    den += phi_prod
                end
            end
            B[k,g] = num/(den*(1-ρ))
        end
    end
end

function Mstep_blockmodel_gauss!(ϕ::Array{Float64, 3}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    Y::Array{Float64, 2}, N::Int, K::Int)
    lv = 0.
    learn_r = 0.9
    cum_den = 0.
    for k in 1:K
        for g in 1:K
            num_gauss = 0.
            num = 0.
            den = 0.
            for j in 1:N
                for i in 1:N
                    phi_prod = ϕ[i,j,k]*ϕ[j,i,g]
                    num += phi_prod*Y[i,j]
                    den += phi_prod
                    num_gauss += phi_prod * (Y[i,j] - B[k,g])^2
                    #lv  += phi_prod * (Y[i,j] - B[k,g])^2
                end
            end
            B[k,g] =  (1-learn_r)*B[k,g] + learn_r*num/(den)
            #cum_den += den
            like_var[k,g] =  (1-learn_r)*like_var[k,g] + learn_r*num_gauss/(den)
        end
    end

    #like_var[1] = lv/cum_den

end

################################################################################
################################################################################

function Mstep_logitNorm_multi!(λ, ν::Vector{Array{Float64, 3}},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, μ, Γ::Array{Float64, 2}, X::Array{Float64, 2}, N::Int, K::Int, P::Int, n_regions::Int)
    a0 = 1.0
    b0 = 1.0

    for k in 1:K
        β_N = 0.5
        for m in 1:3
            σ_2[k] = (b0 + 0.5*sum(Γ[k,:].^2)) / (a0 + 0.5*P)
            S_N = inv(β_N*X*X' + diagm(ones(P)/σ_2[k]))
            Γ[k,:] = β_N * S_N*(X*λ[k,:])
            β_N = N*n_regions/2 * (dot(λ[k,:],λ[k,:])/2 - dot(Γ[k,:], X*λ[k,:]) + 0.5*tr((X * X')*(Γ[k,:]*Γ[k,:]' .+ S_N)) )^(-1)
        end
        #println(β_N)
    end

    μ = Γ * X

    Σ .= zeros(K,K)
    for i_region in 1:n_regions
        for i in 1:N
            Σ .+= (ν[i_region][:,:,i] .+ (λ[:,i+(i_region-1)*N] .- μ[:,i+(i_region-1)*N])*(λ[:,i+(i_region-1)*N] .- μ[:,i+(i_region-1)*N])')
        end
    end
    Σ .= Hermitian(Σ/(N*n_regions))

end

function Mstep_blockmodel_multi!(ϕ::Vector{Array{Float64, 3}}, B::Array{Float64, 2}, ρ::Float64,
    Y::Array{Float64, 2}, N::Int, K::Int, n_regions::Int)
    for k in 1:K
        for g in 1:K
            num = 0.
            den = 0.
            for i_region in 1:n_regions
                for j in 1:N
                    for i in 1:N
                        phi_prod = ϕ[i_region][i,j,k]*ϕ[i_region][j,i,g]
                        num += phi_prod*Y[i,j+(i_region-1)*N]
                        den += phi_prod
                    end
                end
            end
            B[k,g] = num/(den*(1-ρ))
        end
    end
end


function Mstep_blockmodel_gauss_multi!(ϕ::Vector{Array{Float64, 3}}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    Y::Array{Float64, 2}, N::Int, K::Int, n_regions::Int)
    lv = 0.
    learn_r = 0.9
    cum_den = 0.
    for k in 1:K
        for g in 1:K
            num_gauss = 0.
            num = 0.
            den = 0.
            for i_region in 1:n_regions
                for j in 1:N
                    for i in 1:N
                        phi_prod = ϕ[i_region][i,j,k]*ϕ[i_region][j,i,g]
                        num += phi_prod*Y[i,j+(i_region-1)*N]
                        den += phi_prod
                        num_gauss += phi_prod * (Y[i,j+(i_region-1)*N] - B[k,g])^2
                        #lv  += phi_prod * (Y[i,j] - B[k,g])^2
                    end
                end
            end
            B[k,g] =  (1-learn_r)*B[k,g] + learn_r*num/(den)
            #cum_den += den
            like_var[k,g] =  (1-learn_r)*like_var[k,g] + learn_r*num_gauss/(den)
        end
    end

    #like_var[1] = lv/cum_den

end

################################################################################
# function to combine all together to actually run the VEM algorithm

#function run_VEM!(n_iterations::Int, ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
#function run_VEM!(n_iterations::Int, ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
#    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, B::Array{Float64, 2}, ρ::Float64,
#    μ, Y::Array{Float64, 2}, X::Array{Float64, 2},Γ::Array{Float64, 2}, K::Int, N::Int, P::Int)
function run_VEM!(n_iterations::Int, ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, B::Array{Float64, 2}, ρ::Float64,
    μ, Y::Array{Float64, 2}, X::Array{Float64, 2},Γ::Array{Float64, 2}, K::Int, N::Int, P::Int)

    elbows = zeros(n_iterations)
    elbows[1] = ELBO(ϕ, λ, ν, Σ, B, ρ, μ, Y, K, N)
    println(elbows[1])

    for i_iter in 2:n_iterations
        inv_Σ = inv(Σ)
        Estep_logitNorm!(ϕ, λ, ν, inv_Σ, μ, N, K)
        for m in 1:5
        	Estep_multinomial!(ϕ, λ, B, ρ, Y, N, K)
        end
        Estep_logitNorm!(ϕ, λ, ν, inv_Σ, μ, N, K)


        Mstep_logitNorm!(λ, ν, Σ, σ_2, μ, Γ, X, N, K, P)
        Mstep_blockmodel!(ϕ, B, ρ, Y, N, K)
        elbows[i_iter] = ELBO(ϕ, λ, ν, Σ, B, ρ, μ, Y, K, N)
        println("iter num: ", i_iter, " ELBO  ", elbows[i_iter])
        if isnan(elbows[i_iter])
            break
        end
    end

    return elbows
end

# second method for the function run_VEM that has an extra argument n_regins::Int
# that allows to perform the inference if you have more data points
function run_VEM!(n_iterations::Int, ϕ::Vector{Array{Float64, 3}}, λ, ν::Vector{Array{Float64, 3}},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, B::Array{Float64, 2}, ρ::Float64,
    μ, Y::Array{Float64, 2}, X::Array{Float64, 2},Γ::Array{Float64, 2}, K::Int, N::Int, P::Int, n_regions::Int)

    elbows = zeros(n_regions, n_iterations)
    for i_region in 1:n_regions
        elbows[i_region, 1] = ELBO(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, B, ρ, μ[:,(i_region-1)*N+1:i_region*N], Y[:,(i_region-1)*N+1:i_region*N], K, N)
    end
    println(elbows[:,1])

    for i_iter in 2:n_iterations
        inv_Σ = inv(Σ)
        for i_region in 1:n_regions
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), ν[i_region], inv_Σ, @view(μ[:,(i_region-1)*N+1:i_region*N]), N, K)
            ### If you pass slices of matrices to a function in julia, the values of the matrix cannot be modified, you need to use the @view macro
            for m in 1:3
            	Estep_multinomial!(ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), B, ρ, Y[:,(i_region-1)*N+1:i_region*N], N, K)
            end
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), ν[i_region], inv_Σ, @view(μ[:,(i_region-1)*N+1:i_region*N]), N, K)
            Mstep_logitNorm_multi!(λ, ν, Σ, σ_2, μ, Γ, X, N, K, P, n_regions)
            Mstep_blockmodel_multi!(ϕ, B, ρ, Y, N, K, n_regions)
        end


        for i_region in 1:n_regions
            elbows[i_region, i_iter] = ELBO(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, B, ρ, μ[:,(i_region-1)*N+1:i_region*N], Y[:,(i_region-1)*N+1:i_region*N], K, N)
        end
        println("iter num: ", i_iter, " ELBO  \n", elbows[:,i_iter])
        if isnan(elbows[i_iter])
            break
        end
    end


    return elbows
end



###########################################################################

#inference with gaussian matrices

function run_VEM_gauss!(n_iterations::Int, ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    μ, Y::Array{Float64, 2}, X::Array{Float64, 2},Γ::Array{Float64, 2}, K::Int, N::Int, P::Int)

    elbows = zeros(n_iterations)
    elbows[1] = ELBO_gauss(ϕ, λ, ν, Σ, B, μ, Y, K, N, like_var)
    println(elbows[1])

    for i_iter in 2:n_iterations
        inv_Σ = inv(Σ)


        Estep_logitNorm!(ϕ, λ, ν, inv_Σ, μ, N, K)
        for m in 1:5
        	Estep_multinomial_gauss!(ϕ, λ, B, Y, N, K, like_var)
        end


        Estep_logitNorm!(ϕ, λ, ν, inv_Σ, μ, N, K)


        Mstep_logitNorm_flux!(λ, ν, Σ, σ_2, μ, Γ, X, N, K, P)
        Mstep_blockmodel_gauss!(ϕ, B, like_var, Y, N, K)
        elbows[i_iter] = ELBO_gauss(ϕ, λ, ν, Σ, B, μ, Y, K, N, like_var)
        println("iter num: ", i_iter, " ELBO  ", elbows[i_iter])
        println("\n")

        #println(round.(Σ; sigdigits=2), "\n\n")
        if isnan(elbows[i_iter])
            break
        end
    end

    return elbows
end


#########################################################################################


function run_VEM_gauss!(n_iterations::Int, ϕ::Vector{Array{Float64, 3}}, λ, ν::Vector{Array{Float64, 3}},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    μ, Y::Array{Float64, 2}, X::Array{Float64, 2},Γ::Array{Float64, 2}, K::Int, N::Int, P::Int, n_regions::Int)

    elbows = zeros(n_regions, n_iterations)
    #=for i_region in 1:n_regions
        elbows[i_region, 1] = ELBO_gauss(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, σ_2, B, μ[:,(i_region-1)*N+1:i_region*N],  Y[:,(i_region-1)*N+1:i_region*N], K, N, like_var)
        println(i_region, "   ", elbows[i_region,1])
    end
    #println(elbows[:,1])=#

    for i_iter in 1:n_iterations
        inv_Σ = inv(Σ)
        for i_region in 1:n_regions
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), ν[i_region], inv_Σ, @view(μ[:,(i_region-1)*N+1:i_region*N]), N, K)
            for m in 1:5
                Estep_multinomial_gauss!(ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), B, Y[:,(i_region-1)*N+1:i_region*N], N, K, like_var)
            end
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), ν[i_region], inv_Σ, @view(μ[:,(i_region-1)*N+1:i_region*N]), N, K)

            Mstep_logitNorm_flux_multi!(λ, ν, Σ, σ_2, μ, Γ, X, N, K, P, n_regions)
            #Mstep_logitNorm_multi!(λ, ν, Σ, σ_2, μ, Γ, X, N, K, P, n_regions)
            Mstep_blockmodel_gauss_multi!(ϕ, B, like_var, Y, N, K, n_regions)
        end

        #Mstep_logitNorm_flux_multi!(λ, ν, Σ, σ_2, μ, Γ, X, N, K, P, n_regions)
        #Mstep_logitNorm_multi!(λ, ν, Σ, σ_2, μ, Γ, X, N, K, P, n_regions)
        #Mstep_blockmodel_gauss_multi!(ϕ, B, like_var, Y, N, K, n_regions)

        for i_region in 1:n_regions
            elbows[i_region, i_iter] = ELBO_gauss(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, B, μ[:,(i_region-1)*N+1:i_region*N],  Y[:,(i_region-1)*N+1:i_region*N], K, N, like_var)
            println("iter num: ", i_iter, " \t region: ", i_region," ELBO  \n", elbows[i_region,i_iter])
            if isnan(elbows[i_region, i_iter])
                break
            end
        end
    end
    return elbows

end


#########################################################################################


function run_VEM_gauss_NN!(n_iterations::Int, ϕ::Vector{Array{Float64, 3}}, λ, ν::Vector{Array{Float64, 3}},
    Σ::Array{Float64, 2}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    μ, Y::Array{Float64, 2}, X::Array{Float64, 2}, Γ, ps, K::Int, N::Int, P::Int, n_regions::Int)

    elbows = zeros(n_regions, n_iterations)
    opt = ADAM(0.01) #the value in the brackets is
    #################################
    # definition of the loss functional to be used to optimize the flux model
    L(a,b) = (Flux.Losses.kldivergence(softmax(Γ(a)), softmax(b)))

    #################################


    for i_iter in 1:n_iterations
        inv_Σ = inv(Σ)
        for i_region in 1:n_regions
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), ν[i_region], inv_Σ, Float64.(μ[:,(i_region-1)*N+1:i_region*N]), N, K)
            for m in 1:5
                Estep_multinomial_gauss!(ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), B, Y[:,(i_region-1)*N+1:i_region*N], N, K, like_var)
            end
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,(i_region-1)*N+1:i_region*N]), ν[i_region], inv_Σ, (μ[:,(i_region-1)*N+1:i_region*N]), N, K)
            n_flux = 15
            for i_flux in 1:n_flux
                gs = gradient(()-> L(Float32.(X), Float32.(λ)), ps)
                Flux.Optimise.update!(opt, ps, gs)
            end

            μ = Γ(X);

        end



        #=Σ .= zeros(K,K)
        for i_region in 1:n_regions
            for i in 1:N
                Σ .+= 1/(N*n_regions) * (ν[i_region][:,:,i] .+ (λ[:,i+(i_region-1)*N] .- μ[:,i+(i_region-1)*N])*(λ[:,i+(i_region-1)*N] .- μ[:,i+(i_region-1)*N])')
            end
        end
        Σ .= Hermitian(Σ)=#

        Mstep_blockmodel_gauss_multi!(ϕ, B, like_var, Y, N, K, n_regions)

        for i_region in 1:n_regions
            elbows[i_region, i_iter] = ELBO_gauss(ϕ[i_region], λ[:,(i_region-1)*N+1:i_region*N], ν[i_region], Σ, B, μ[:,(i_region-1)*N+1:i_region*N],  Y[:,(i_region-1)*N+1:i_region*N], K, N, like_var)
            println("iter num: ", i_iter, " \t region: ", i_region," ELBO  \n", elbows[i_region,i_iter])
            if isnan(elbows[i_region, i_iter])
                break
            end
        end
    end
    return elbows

end

##########################################################################################

#inference with gaussian matrices but with a different update scheduling to "equilibrate"
# to non-random initialization for Gamma

function run_VEM_gauss_2!(n_iterations::Int, ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, σ_2::Array{Float64, 1}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    μ, Y::Array{Float64, 2}, X::Array{Float64, 2},Γ::Array{Float64, 2}, K::Int, N::Int, P::Int)

    elbows = zeros(n_iterations)
    elbows[1] = ELBO_gauss(ϕ, λ, ν, Σ, B, μ, Y, K, N, like_var)
    println(elbows[1])

    for i_iter in 1:n_iterations
        inv_Σ = inv(Σ)


        Estep_logitNorm!(ϕ, λ, ν, inv_Σ, μ, N, K)
        for m in 1:5
        	Estep_multinomial_gauss!(ϕ, λ, B, Y, N, K, like_var)
        end


        Estep_logitNorm!(ϕ, λ, ν, inv_Σ, μ, N, K)

        if (i_iter > 30)
            for m in 1:5
                Mstep_logitNorm!(λ, ν, Σ, σ_2, μ, Γ, X, N, K, P)
                Estep_logitNorm!(ϕ, λ, ν, inv_Σ, μ, N, K)
            end
        end
        if (i_iter < 30)
            for m in 1:5
                Σ .= zeros(K,K)
                for i in 1:N
                    Σ .+= 1/N * (ν[:,:,i] .+ (λ[:,i] .- μ)*(λ[:,i] .- μ)')
                end
                Σ .= Hermitian(Σ)

                Estep_logitNorm!(ϕ, λ, ν, inv_Σ, μ, N, K)
            end
        end

        Mstep_blockmodel_gauss!(ϕ, B, like_var, Y, N, K)
        elbows[i_iter] = ELBO_gauss(ϕ, λ, ν, Σ, B, μ, Y, K, N, like_var)
        println("iter num: ", i_iter, " ELBO  ", elbows[i_iter])
        println("\n")

        if isnan(elbows[i_iter])
            break
        end
    end

    return elbows
end



#inference with gaussian matrices BUT NO COVARIATES

function run_VEM_gauss!(n_iterations::Int, ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    μ, Y::Array{Float64, 2}, K::Int, N::Int)

    elbows = zeros(n_iterations)
    elbows[1] = ELBO_gauss(ϕ, λ, ν, Σ, B, μ, Y, K, N, like_var)
    println(elbows[1])

    for i_iter in 2:n_iterations
        inv_Σ = inv(Σ)


        Estep_logitNorm_noX!(ϕ, λ, ν, inv_Σ, μ, N, K)
        for m in 1:5
        	Estep_multinomial_gauss!(ϕ, λ, B, Y, N, K, like_var)
        end


        Estep_logitNorm_noX!(ϕ, λ, ν, inv_Σ, μ, N, K)


        Mstep_logitNorm_noX!(λ, ν, Σ, μ, N, K)
        Mstep_blockmodel_gauss!(ϕ, B, like_var, Y, N, K)
        elbows[i_iter] = ELBO_gauss(ϕ, λ, ν, Σ, B, μ, Y, K, N, like_var)
        println("iter num: ", i_iter, " ELBO  ", elbows[i_iter])
        println("\n")
        if isnan(elbows[i_iter])
            break
        end
    end

    return elbows
end
