
#module PFParallelChainMDP
using Random # for AbstractRNG
using POMDPModelTools
using POMDPs
using POMDPModels
using StaticArrays
using Distributions
using POMDPPolicies
using POMDPSimulators
using Statistics

const Vec2 = SVector{2, Int64}

    mutable struct PParallelChainMDP <: MDP{Vec2, Int64}
        len::Int64
        num_chains::Int64
        disc::Float64
        Rs::Array{Float64}
        # Rs_stds::Array{Float64}
        # rng::MersenneTwister
    end
    function POMDPs.generate_s(p::PParallelChainMDP, s::Vec2, a::Int64, rng::AbstractRNG)
        if s[2] != 1
            return Vec2(s[1], s[2] + 1)
        end
        if s[2] == 1
            return Vec2(a, 2)
        end
    end

    # function POMDPs.reward(p::PParallelChainMDP, s::Vec2, a::Int64)
    #     # println("s:$(s),len:$(p.len)")
    #     if s[2] == p.len -1
    #         return p.Rs[s[1]] # Should be random
    #     end
    #     return 0
    # end
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
            reward = p.Rs[s[1]] # Should be random
            # println("reward: $reward")
            return reward
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
      # println("s:$s")
      # println("li:$(LinearIndices((p.num_chains, p.len))[s...])")
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
        # println("states:$(vec( [Vec2(x,y) for x in 1:p.num_chains, y in 1:p.len]))")
        vec( [Vec2(x,y) for x in 1:p.num_chains, y in 1:p.len])
    end
    function POMDPs.actions(p::PParallelChainMDP)
       return 1:p.num_chains
    end



curry(f, x) = (xs...) -> f(x, xs...)
function run_chain!(;mdp_iter_builder, true_mdp, do_update_priors, update_priors, priors, true_vals,
                     n_agents, num_states, num_chains,
                     epochs, steps, is_thompson_sampling, rng) #, rev_action_map)
    r_history =[]
    # N_lists =
    latest_priors = deepcopy(priors)
    ap = Poisson(1)
    for e in 1:epochs
        agents = []
        agents_done = zeros(Bool, 1, n_agents)
        done = false
        t = 0
        started = 0
        while ! done
           done = true
           max_finished = 1
           # i = max_finished

           # println("$(n_agents - i)")
           # println("start_add:$start_add")
           if started < n_agents
               start_add = rand(rng, ap)
               if start_add > 0
                 for j in 1:start_add
                   if length(agents) < n_agents
                     started += 1
                     push!(agents, mdp_iter_builder(rng, true_mdp, latest_priors, started, num_states, num_chains, steps, nothing))
                   end
                 end
               end
           end
           for i in max_finished:started
                # if i >= start
                #     start += 1
                #     done = false
                #     break
                # end
                if isempty(agents[i])
                    # println("agent $i is done")
                    continue
                end
                res = popfirst!(agents[i])
                r = res[:r]
                t = res[:t]
                st = res[:s]
                sp = res[:sp]
                # note sp is ignored
                r = POMDPs.reward(true_mdp, st, 1, Vec2(1,1)) != 0 ? randn(rng, Float32,1)[1]  + POMDPs.reward(true_mdp, st, 1, Vec2(1,1)) : 0

                push!(r_history, (e,i,t,st,r))
                li = LinearIndices((num_chains, num_states))
                # N_lists[i][s]] += 1
                if i ==1 || i % floor(n_agents/10) == 0
                   println("e: $e, t: $t, agent $i, result: $res")
                end
                if isempty(agents[i]) && ! agents_done[i]
                    # println("agent $i is done")
                    agents_done[i] = true
                    # println("Updating priors")
                    latest_priors = update_priors(priors, r_history)
                    max_finished = i
                end
                # for thompson sampling, we need to call the builder again, if the agent isn't done.
                if is_thompson_sampling && ! agents_done[i]
                  agents[i] = mdp_iter_builder(rng, true_mdp, latest_priors, i, num_states, num_chains, steps, sp)
                end
           end
           for i in 1:n_agents

                if  length(agents) < i || ( ! isempty(agents[i]))

                    done = false
                end
           end
        end
    end
    return r_history
end

function setup_agents(states, num_states, num_agents, actions, num_actions, policy_function)
    Q_tables = []
    policies = []
    println("setup agents")
    empty_N = zeros( num_states+2, num_actions)
    N_tables = Dict{Int32, typeof(empty_N)}()
    Central_Q_table = Dict{Int32, Dict{Int32, Float32}}()

    for i in 1:num_agents
        N_tables[i] = deepcopy(empty_N)
        push!(policies, curry(curry(curry(curry(policy_function, q_mat), N_tables),i), actions))
    end
    return (Q_tables, N_tables, policies)
end
