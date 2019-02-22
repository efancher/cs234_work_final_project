
module PFChainMDP
using Random # for AbstractRNG
using POMDPModelTools
using POMDPPolicies
using POMDPSimulators
using POMDPs
    export PChainMDP
    struct PChainMDP <: MDP{Int, Symbol}
        len::Int
        p_success::Float64
        discount::Float64
        theta::Int
    end
    function POMDPs.generate_s(p::PChainMDP, s::Int, a::Symbol, rng::AbstractRNG)
        if a == :right
            success = min(s+1, p.len)
            failure = max(s-1, 1)
        else # a == :left
            success = max(s-1, 1)
            failure = min(s+1, p.len)
        end
        if s + 1 == p.len
            return p.len
        elseif  s == 2
            return 1
        end
        return rand(rng) < p.p_success ? success : failure
    end

    function POMDPs.reward(p::PChainMDP, s::Int, a::Symbol)
        if s == 2
            return p.theta
        end
        if s + 1 == p.len
            return -p.theta
        end
        if s == 0 || s == p.len
            return 0
        end
        return -1
    end

    function POMDPs.isterminal(p::PChainMDP, s::Int)
        if s == 1
            return true
        end
        if s == p.len
            return true
        end
        return false
    end

    function POMDPs.initialstate_distribution(p::PChainMDP)
        return Deterministic(Int64((p.len+2)/2))
    end

end
