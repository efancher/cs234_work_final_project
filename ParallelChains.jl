
#module PFParallelChainMDP
using Random # for AbstractRNG
using POMDPModelTools
using POMDPs
using POMDPModels
using StaticArrays
using Distributions

const Vec2 = SVector{2, Int64}

    struct PParallelChainMDP <: MDP{Vec2, Int64}
        len::Int64
        num_chains::Int64
        disc::Float64
        Rs::Array{Float64}
    end
    function POMDPs.generate_s(p::PParallelChainMDP, s::Vec2, a::Int64, rng::AbstractRNG)
        if s[2] != 1
            return Vec2(s[1], s[2] + 1)
        end
        if s[2] == 1
            return Vec2(a, 2)
        end
    end

    function POMDPs.reward(p::PParallelChainMDP, s::Vec2, a::Int64)
        println("s:$(s),len:$(p.len)")
        if s[2] == p.len -1
            return p.Rs[s[1]] # Should be random
        end
        return 0
    end
    function POMDPs.isterminal(p::PParallelChainMDP, s::Vec2)
        if s[2] == p.len
            return true
        end
        return false
    end

    function POMDPs.initialstate(p::PParallelChainMDP, rng::AbstractRNG)
        return Vec2(1,1)
    end

    function POMDPs.discount(p::PParallelChainMDP)
        return p.disc
    end

    function POMDPs.n_states(p::PParallelChainMDP)
        return p.num_chains * p.len
    end

    function POMDPs.n_actions(p::PParallelChainMDP)
        return p.num_chains
    end


    function POMDPs.reward(p::PParallelChainMDP, s::Vec2, a::Int64, sp::Vec2)
        if s[2] == p.len -1
            return p.Rs[s[1]] # Should be random
        end
        return 0
    end

    function POMDPs.transition(p::PParallelChainMDP, s::Vec2, a::Int64)
       if s[2] != 1
            return Deterministic(Vec2(s[1], s[2] + 1))
        end
        if s[2] == 1
            return Deterministic(Vec2(a, 2))
        end
    end
    function POMDPs.pdf(d::SArray, sp::SArray)
        return d.p[sp]
    end
    function POMDPs.stateindex(p::PParallelChainMDP, s::Vec2)
      println("s:$s")
      println("li:$(LinearIndices((p.num_chains, p.len))[s...])")
      return LinearIndices((p.num_chains, p.len))[s...]
    end

    function POMDPs.actionindex(p::PParallelChainMDP, a::Int64)
      return a
    end
    # actionindex(::MDP, ::Action)
    # actions(::MDP, ::State)
    function POMDPs.actions(p::PParallelChainMDP, s::Vec2)
       if s[2] ==  1
         return 1:p.num_chains
       end
       return [1]
    end
    # support(::StateDistribution)
    # pdf(::StateDistribution, ::State)

    # states(::MDP)
    # actions(::MDP)
    function POMDPs.states(p::PParallelChainMDP)
        println("states:$(vec( [Vec2(x,y) for x in 1:p.num_chains, y in 1:p.len]))")
        vec( [Vec2(x,y) for x in 1:p.num_chains, y in 1:p.len])
    end
    function POMDPs.actions(p::PParallelChainMDP)
       return 1:p.num_chains
    end
