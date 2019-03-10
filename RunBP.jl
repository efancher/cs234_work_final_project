using POMDPs
using Random # for AbstractRNG
using POMDPModelTools
using Pkg

using CSV
using DataFrames
Pkg.add("JSON")

POMDPs.add_registry()
using Pkg; Pkg.add("DiscreteValueIteration")

using DiscreteValueIteration

using Pkg
Pkg.add("StaticArrays")
using POMDPSimulators
using POMDPPolicies

include("./ChainMDP2.jl")

function ucb_pol(li, policy, priors, i, actions, s)
    best_action = action(policy, s)
    #println("in state: $s, best action:$best_action")
    return action(policy, s)
end



using DataFrames
using Statistics




curry(f, x) = (xs...) -> f(x, xs...)
function ucb_mdp_builder(rng, true_mdp, priors, i, num_states, num_chains, steps, start_state)
  # build mdp
  new_means = priors.theta + priors.std
  mdp = PParallelChainMDP(num_states+1,num_chains, .9,
        [priors.theta[j] + 4* priors.std[j] for j in 1:num_chains])
  solver = ValueIterationSolver(max_iterations=1000, belres=1e-6, include_Q=true)#, verbose=true) # initializes the Solver type
  vip = solve(solver, mdp)
  li = LinearIndices((num_chains, num_states))
  new_policy = FunctionPolicy(curry(curry(curry(curry(curry(ucb_pol, li), vip), priors),i), actions))
  if start_state == nothing
    return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, "s,a,r,sp,t",
                              max_steps=steps))
  else
     return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, start_state, "s,a,r,sp,t",
                              max_steps=steps))
  end
end



include("./ParallelChains.jl")
using Distributions
function do_runs(nruns, epochs, num_agents, num_chains, num_states, H, update_priors_check, update_priors,
                 mdp_iter_builder, is_thompson_sampling, base, rng)
    # UCB

    # setup constants here

    # So best idea:
    # for ucb:
    # we generate an mdp with mu + sigma rewards (priors) for each end node.
    #  we collect history until the criteria in PAC-EXPLORE then update the prior
    #
    # for thompson sampling
    # at each time step (might skip this for this problem and save for max rew path)
    #   take rewards and use to create a posterior.
    # for seed sampling
    # at beginning of episode
    #   each agent generates a new random "seed"
    #   at each time step
    #     each agent generates a new mdp based on a deterministic mapping from seed to rewards (which also takes
    #              history into account)
    # this needs to update the actual reward.

    ## PC notes
    ## theta_c ~ N(0, 100 + c)
    r_mean = 0
    r_std = 1
    ## sigma is 1
    ## Reward is noisy ~ N(theta_c, sigma^2)
    ## Initial prior mean is N(theta_c, sigma^2 + c)
    runs = []
    true_values_list = DataFrame( run = [],
                                 c = [],
                                 theta = [],
                                 std = [])
    for run in 1:nruns
        th_c = [randn(rng, Float64, 1)[1] * sqrt(base + i) for i in 1:num_chains]
        #th_c[1] = 300
        println(th_c)
        th_c_sd = sqrt(var(th_c))
        r_std = th_c_sd
        true_values = DataFrame( run = [run for i in 1:num_chains],
                                 c = [i for i in 1:num_chains],
                                 theta = th_c,
                                 std = [r_std  for i in 1:num_chains])
        print("true mdp: $true_values")
        #priors = DataFrame(theta = [th_c[i] for i in 1:num_chains],
        priors = DataFrame(theta = [0 for i in 1:num_chains],
                           std = [(sqrt(r_std^2) + i) for i in 1:num_chains],
                           state = [(chain,num_states) for chain in 1:num_chains],
                           c = [i for i in 1:num_chains])
        print("priors: $priors")
        mdp = PParallelChainMDP(num_states+1,num_chains, .9,
                true_values.theta)
        do_update_priors() =  false
        # priors = randn(rng, Float32, num_chains) .* (r_std + c) .+ r_mean
        hist = run_chain!(
                   mdp_iter_builder=mdp_iter_builder,
                   true_mdp=mdp,
                   do_update_priors=update_priors_check,
                   update_priors=update_priors,
                   priors=priors,
                   true_vals=true_values,
                   n_agents=num_agents,
                   num_states=num_states,
                   num_chains=num_chains,
                   epochs=epochs,
                   steps=H,
                   is_thompson_sampling=is_thompson_sampling,
                   rng=rng) #,
                   #rev_action_map=rev_action_map)
        #print(Q_tables)
        println("(e,i,t,st,r)")
        hist
        for (e, ag, t, st, r)  in hist
          push!(runs, (run, e, ag, t, st, r))
        end
        append!(true_values_list, true_values)

    end
    return (true_values_list, runs)
end



using DataFrames
using Statistics
function ucb_update_priors(priors, hist, true_vals)
    # each state must be visted me times to update the prior
    # me is # of S*D^2 s= states, D =diameter (https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9542/9919)
    #     (D defined in http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf)
    ## ^^^ Actually, think should just do this statistically, and if n > 2, then we can adjust the prior.
    df = DataFrame(epoch = [x[1] for x in hist], agent=[x[2] for x in hist],
               time = [x[3] for x in hist], chain = [x[4][1] for x in hist], chain_state = [x[4][2] for x in hist],
        reward = [x[5] for x in hist])
    new_theta = []
    new_std = []
    new_state = []
    for arow in eachrow(priors)
        if nrow(df[(df.chain.==arow.state[1]).&(df.chain_state.==arow.state[2]),:]) > 1
            mu_p = arow.theta
            true_std = true_vals[true_vals.c.==arow.c,:].std[1]
            s_p_2 = arow.std^2
            mu_s = mean(df[(df.chain.==arow.state[1]).&(df.chain_state.==arow.state[2]),:].reward)
            s_s_2 = var(df[(df.chain.==arow.state[1]).&(df.chain_state.==arow.state[2]),:].reward)
            cnt = nrow(df[(df.chain.==arow.state[1]).&(df.chain_state.==arow.state[2]),:])
            #tmp_theta =  (s_p_2/(s_s_2 + s_p_2)) * mu_s + (s_s_2/(s_s_2 + s_p_2)) * mu_p
            #tmp_std = 1/(1/s_p_2 + 1/s_s_2)
            tmp_theta =  (s_p_2/(true_std/cnt + s_p_2)) * mu_s + (s_s_2/(true_std/cnt + s_p_2)) * mu_p
            tmp_std = 1/(1/s_p_2 + cnt/true_std)
        else
            tmp_theta =  arow.theta
            tmp_std = arow.std
        end
        push!(new_state, arow.state)
        push!(new_theta, tmp_theta)
        push!(new_std, tmp_std)
    end
    new_priors = DataFrame(theta=new_theta, std=new_std, state=new_state)
    return new_priors
end


function get_average_regret(results, chain_len, num_chains, theta)
    df = DataFrame(run = [x[1] for x in results], epoch = [x[2] for x in results], agent=[x[3] for x in results],
               time = [x[4] for x in results], c = [x[5][1] for x in results], state = [x[5][2] for x in results],
               reward = [x[6] for x in results])
    df_R = df[df.reward.!==0,:]
    th_max = by(theta, :run, :theta => maximum)
    th_min = by(theta, :run, :theta => minimum)
    theta_joined = join(join(theta, th_max, on = :run), th_min, on = :run)
    df_joined = join(df_R, theta_joined, on = [:run,:c] )
    # needs to be max theta
    df_joined.Regret = df_joined.theta_maximum - df_joined.reward
    df_joined
end


curry(f, x) = (xs...) -> f(x, xs...)
function ssgn_mdp_builder(rng, true_mdp, priors, i, num_states, num_chains, steps, start_state)
  # build mdp
  # new_means = priors.theta + priors.std
  mdp = PParallelChainMDP(num_states+1,num_chains, .9,
        #[randn(rng, Float32,1)[1] *  1 + priors.theta[j] for j in 1:num_chains])
        [randn(rng, Float32,1)[1] *  (2*priors.std[j]) + priors.theta[j] for j in 1:num_chains])
  solver = ValueIterationSolver(max_iterations=1000, belres=1e-6, include_Q=true)#, verbose=true) # initializes the Solver type
  vip = solve(solver, mdp)
  li = LinearIndices((num_chains, num_states))
  new_policy = FunctionPolicy(curry(curry(curry(curry(curry(ucb_pol, li), vip), priors),i), actions))
  if start_state == nothing
    return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, "s,a,r,sp,t",
                              max_steps=steps))
  else
     return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, start_state, "s,a,r,sp,t",
                              max_steps=steps))
  end
end



# Full loop
function run_all()
    full_df = nothing;
    base = 100

    rng = MersenneTwister()
    for num_agents in [10000]
        println("*****************number of agents:$num_agents*******************")
        num_chains = 10
        num_states = 5
        epochs = 1
        nruns = 15
        H = 12
        do_update_priors() = true
        for run_type in ["Thompson Sampling", "Seed Sampling", "UCB"]
            println("------------------Run Type:$run_type------------------------")
            if run_type == "Thompson Sampling"
                true_values_list, runs = do_runs(nruns, epochs, num_agents, num_chains, num_states, H,
                                                 do_update_priors, ucb_update_priors,  ssgn_mdp_builder, true,
                                                 base, rng)
            elseif run_type == "UCB"
                true_values_list, runs = do_runs(nruns, epochs, num_agents, num_chains, num_states, H,
                                                 do_update_priors, ucb_update_priors, ucb_mdp_builder, false,
                                                 base, rng)
            else
                true_values_list, runs = do_runs(nruns, epochs, num_agents, num_chains, num_states, H,
                                                 do_update_priors, ucb_update_priors,  ssgn_mdp_builder, false,
                                                 base, rng)

            end

            chain_len = num_states
            num_chains = num_chains
            regret = get_average_regret(runs, chain_len, num_chains, true_values_list)
            regret.name = [run_type for i in 1:nrow(regret)]
            regret.num_agents = [num_agents for i in 1:nrow(regret)]
            regret.base = [base for i in 1:nrow(regret)]
            if full_df == nothing
                full_df = regret
            else
                append!(full_df,regret)
            end
        end
    end

    println("***********************Done!!!*********************************")
    full_df[full_df[:Regret] .< 0.0,:Regret] = 0.0
    CSV.write("pc_simulation_10000.csv", full_df)
    return full_df
end
