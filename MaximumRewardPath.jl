
#module PFMaxRewardPathMDP
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

    mutable struct PMaxRewardPathMDP <: MDP{Vec2, Int64}
        len::Int64
        disc::Float64
        Rs::Array{Float64,2}
        trans::Dict{Int64, Array{Int64}}
        sigma::Float64
        # Rs_stds::Array{Float64}
        # rng::MersenneTwister
    end
    # function POMDPs.generate_s(p::PMaxRewardPathMDP, s::Vec2, a::Int64, rng::AbstractRNG)
    #     if s[2] != 1
    #         return Vec2(s[1], s[2] + 1)
    #     end
    #     if s[2] == 1
    #         return Vec2(a, 2)
    #     end
    # end

    # function POMDPs.reward(p::PMaxRewardPathMDP, s::Vec2, a::Int64)
    #     # println("s:$(s),len:$(p.len)")
    #     if s[2] == p.len -1
    #         return p.Rs[s[1]] # Should be random
    #     end
    #     return 0
    # end
    function POMDPs.isterminal(p::PMaxRewardPathMDP, s::Int64)
        # if s[2] == p.len
        #     return true
        # end
        return false
    end

    function POMDPs.initialstate(p::PMaxRewardPathMDP, rng::AbstractRNG)
        return Vec2(1,1)
    end

    function POMDPs.discount(p::PMaxRewardPathMDP)
        return p.disc
    end

    function POMDPs.n_states(p::PMaxRewardPathMDP)
        return p.len
    end

    function POMDPs.n_actions(p::PMaxRewardPathMDP)
        return p.len
    end
    function POMDPs.reward(p::PMaxRewardPathMDP, s::Int64, a::Int64, sp::Int64)
        return POMDPs.reward(p, Vec2(1,s), a, Vec2(1,1))

    end
    # function POMDPs.reward(p::PMaxRewardPathMDP, s::Vec2, a::Int64)
    #     return POMDPs.reward(p, s, a, Vec2(1,1))
    #
    # end
    function POMDPs.reward(p::PMaxRewardPathMDP, s::Vec2, a::Int64, sp::Vec2)
        # println("Reward:for s[2]:$(s[2]),a:$a, is $(p.Rs[s[2], a])")
        reward = p.Rs[s[2], a] # Should be random
        # println("reward: $reward")
        return reward

    end

    function POMDPs.transition(p::PMaxRewardPathMDP, s::Vec2, a::Int64)
        #println("transition: $s, $a")
          return Deterministic(Vec2(1,a))
    end

    function POMDPs.pdf(d::SArray, sp::SArray)
        #println("pdf")
        return d.p[sp]
    end
    function POMDPs.stateindex(p::PMaxRewardPathMDP, s::Vec2)
      # println("s:$s")
      # println("li:$(LinearIndices((p.num_chains, p.len))[s...])")
      return s[2]
    end

    function POMDPs.actionindex(p::PMaxRewardPathMDP, a::Int64)
      #println("actionindex")
      return a
    end
    # actionindex(::MDP, ::Action)
    # actions(::MDP, ::State)
    function POMDPs.actions(p::PMaxRewardPathMDP, s::Vec2)
       #println("actions: $s")
       #println("in state $s, can take action $(p.trans[s[2]])")
       return p.trans[s[2]]
    end
    # support(::StateDistribution)
    # pdf(::StateDistribution, ::State)

    # states(::MDP)
    # actions(::MDP)
    function POMDPs.states(p::PMaxRewardPathMDP)
        #println("states")
        # println("states:$(vec( [Vec2(x,y) for x in 1:p.num_chains, y in 1:p.len]))")
        vec( [Vec2(1,y) for  y in 1:p.len])
    end
    function POMDPs.actions(p::PMaxRewardPathMDP)
       #println("actions")
       return 1:p.len
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
            agent_steps = []
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
                         push!(agent_steps, steps)
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
                    agent_steps[i] -= 1
                    r = res[:r]
                    t = res[:t]
                    st = res[:s]
                    sp = res[:sp]
                    a = res[:a]
                    # note sp is ignored
                    # e − 0.005, 0.01
                    r = POMDPs.reward(true_mdp, st, a, sp) != 0 ? exp(randn(rng, Float32,1)[1]*true_mdp.sigma  + 
                                                (log(POMDPs.reward(true_mdp, st, a, sp)) - (true_mdp.sigma)^2 )): 0

                    push!(r_history, (e,i,t,st[2],sp[2],a, r))
                    # N_lists[i][s]] += 1
                    #if i ==1 || i % floor(n_agents/10) == 0
                       println("e: $e, t: $t, agent $i, actual reward: $r, result: $res")
                    #end
                    if isempty(agents[i]) && ! agents_done[i]
                        # println("agent $i is done")
                        agents_done[i] = true
                        # println("Updating priors")
                        latest_priors = update_priors(priors, r_history, true_vals, true_mdp)
                        max_finished = i
                    end
                    # for thompson sampling, we need to call the builder again, if the agent isn't done.
                    if is_thompson_sampling && ! agents_done[i]
                      agents[i] = mdp_iter_builder(rng, true_mdp, latest_priors, i, num_states, num_chains, agent_steps[i], sp)
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
