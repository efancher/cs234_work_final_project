using POMDPs
using Random # for AbstractRNG
using POMDPModelTools
using Pkg

using DataFrames
Pkg.add("JSON")

POMDPs.add_registry()
using Pkg; Pkg.add("DiscreteValueIteration")

using DiscreteValueIteration
using Pkg
Pkg.add("StaticArrays")
using POMDPSimulators
using POMDPPolicies

using DataFrames
using Statistics

using CSV


include("./MaximumRewardPath.jl")

function build_random_path(g_len)
  trans = Dict{Int64, Array{Int64}}()
  for i in 1:g_len

        for j in 1:g_len
          # print(rand(Bernoulli(0.2)))
          p = 2 * log(g_len)/g_len
          # print("p:$p")
          if rand(Bernoulli(p)) == 1
                if !(i  in keys(trans))
                    trans[i] = []
                end
                append!(trans[i], j)
          end
        end
  end
  # print(trans)

  is = keys(trans)
  Rs = zeros(Float64,g_len,g_len)
  for i in is
    for j in trans[i]
      Rs[i,j] = exp(randn(Float64)*4)
    end
  end
  new_mdp = PMaxRewardPathMDP(g_len, .9,
        Rs,
        trans)
  #print(new_mdp)
  #return(new_mdp)
 end


function ucb_pol(li, policy, priors, i, actions, s)
    best_action = action(policy, s)
    #println("in state: $s, best action:$best_action")
    return action(policy, s)
end


curry(f, x) = (xs...) -> f(x, xs...)
function ucb_mdp_builder(rng, true_mdp, priors, i, num_states, num_chains, steps, start_state)
  # build mdp
  new_means = priors.theta + priors.std
  mdp = deepcopy(true_mdp)
  # Should use theta and std

  li = LinearIndices((num_states, num_states))
  mdp.Rs = reshape([prior.theta + 6* prior.std for prior in eachrow(priors)],num_states,num_states)
  # println("Rs: $(mdp.Rs)")
  # println("prior: $(priors.theta)")
  #println("Solving")
  solver = SparseValueIterationSolver(max_iterations=1000, belres=1e-6, include_Q=true)#, verbose=true) # initializes the Solver type
  vip = solve(solver, mdp)
  new_policy = FunctionPolicy(curry(curry(curry(curry(curry(ucb_pol, 1), vip), priors),i), actions))
  if start_state == nothing
    return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, "s,a,r,sp,t",
                              max_steps=steps))
  else
     return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, start_state, "s,a,r,sp,t",
                              max_steps=steps))
  end
end

curry(f, x) = (xs...) -> f(x, xs...)
function ucb_mdp_builder(rng, true_mdp, priors, i, num_states, num_chains, steps, start_state)
  # build mdp
  new_means = priors.theta + priors.std
  mdp = deepcopy(true_mdp)
  # Should use theta and std

  li = LinearIndices((num_states, num_states))
  mdp.Rs = reshape([prior.theta + 6* prior.std for prior in eachrow(priors)],num_states,num_states)
  # println("Rs: $(mdp.Rs)")
  # println("prior: $(priors.theta)")
  #println("Solving")
  solver = SparseValueIterationSolver(max_iterations=1000, belres=1e-6, include_Q=true)#, verbose=true) # initializes the Solver type
  vip = solve(solver, mdp)
  new_policy = FunctionPolicy(curry(curry(curry(curry(curry(ucb_pol, 1), vip), priors),i), actions))
  if start_state == nothing
    return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, "s,a,r,sp,t",
                              max_steps=steps))
  else
     return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, start_state, "s,a,r,sp,t",
                              max_steps=steps))
  end
end

curry(f, x) = (xs...) -> f(x, xs...)
function ssgn_mdp_builder(rng, true_mdp, priors, i, num_states, num_chains, steps, start_state)
  # build mdp
  # new_means = priors.theta + priors.std
  mdp = deepcopy(true_mdp)

  li = LinearIndices((num_states, num_states))
  mdp.Rs = reshape([randn(rng, Float32) * (2*prior.std) + prior.theta  for prior in eachrow(priors)],num_states,num_states)
  #println("Solving")
  solver = SparseValueIterationSolver(max_iterations=1000, belres=1e-6, include_Q=true)#, verbose=true) # initializes the Solver type
  vip = solve(solver, mdp)
  #li = LinearIndices((num_chains, num_states))
  new_policy = FunctionPolicy(curry(curry(curry(curry(curry(ucb_pol, 1), vip), priors),i), actions))
  if start_state == nothing
    return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, "s,a,r,sp,t",
                              max_steps=steps))
  else
     return Iterators.Stateful(stepthrough(deepcopy(true_mdp), new_policy, start_state, "s,a,r,sp,t",
                              max_steps=steps))
  end
end


function ucb_update_priors(priors, hist, true_vals, true_mdp)
    # each state must be visted me times to update the prior
    # me is # of S*D^2 s= states, D =diameter (https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9542/9919)
    #     (D defined in http://www.jmlr.org/papers/volume11/jaksch10a/jaksch10a.pdf)
    ## ^^^ Actually, think should just do this statistically, and if n > 2, then we can adjust the prior.
    #println(hist)
    #Any[(1, 1, 1, 1, 1.79837, 2), (1, 1, 2, 2, 0, 1), (1, 1, 3, 1, 2.04313, 2), (1, 1, 4, 2, 0, 1), (1, 1, 5, 1, -0.693505, 2)]
    #println("ucb_update_priors")
    #println(hist[1])
    df = DataFrame(epoch = [x[1] for x in hist], agent=[x[2] for x in hist],
               time = [x[3] for x in hist], state = [x[4] for x in hist], state2 = [x[5] for x in hist], a = [x[6] for x in hist],
        reward = [x[7] for x in hist])
        
    #println(df[1,:])
    new_theta = []
    new_std = []
    new_c = []
    cs = CartesianIndices(size(true_mdp.Rs))
    #println("cs:$([(idx, val) for  (idx,val) in enumerate(cs)])")
    for arow in eachrow(priors)
        if nrow(df[(df.state.==cs[arow.c][1]).&(df.state2.==cs[arow.c][1]),:]) > 1
            mu_p = arow.theta
            true_std = true_vals[true_vals.c.==arow.c,:].std[1]
            s_p_2 = arow.std^2
            mu_s = mean(df[(df.state.==cs[arow.c][1]).&(df.state2.==cs[arow.c][1]),:].reward)
            s_s_2 = var(df[(df.state.==cs[arow.c][1]).&(df.state2.==cs[arow.c][1]),:].reward)
            cnt = nrow(df[(df.state.==cs[arow.c][1]).&(df.state2.==cs[arow.c][1]),:])
            #tmp_theta =  (s_p_2/(s_s_2 + s_p_2)) * mu_s + (s_s_2/(s_s_2 + s_p_2)) * mu_p
            #tmp_std = 1/(1/s_p_2 + 1/s_s_2)
            tmp_theta =  (s_p_2/(true_std/cnt + s_p_2)) * mu_s + (s_s_2/(true_std/cnt + s_p_2)) * mu_p
            tmp_std = 1/(1/s_p_2 + cnt/true_std)
        else
            tmp_theta =  arow.theta
            tmp_std = arow.std
        end
        push!(new_c, arow.c)
        push!(new_theta, tmp_theta)
        push!(new_std, tmp_std)
    end
    new_priors = DataFrame(theta=new_theta, std=new_std, c=new_c)
    #println("posterior: $new_priors")
    return new_priors
end

using Distributions
function do_runs(nruns, epochs, num_agents,  num_states, H, update_priors_check, update_priors,
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
        #println("num_states: $num_states")
        mdp = build_random_path(num_states)
        li = LinearIndices((num_states, num_states))
        cs = CartesianIndices(size(mdp.Rs))
        #println(li)
        #println(cs)
        th_c = reshape(mdp.Rs, 1,:)
        #println("rewards")
        #println(mdp.Rs)
        #println("thetas")
        #println(th_c)
        for (idx, val) in enumerate(th_c)
          println("$idx, $val, $(mdp.Rs[cs[idx]])")
        end
        # break
        #th_c[1] = 300
        #print("th_c:::::")
        #println(th_c)
        th_c_sd = sqrt(var(th_c))
        r_std = th_c_sd
        num_rewards = length(th_c)
        th_c = dropdims(th_c;dims=1)
        state_pairs = [(i,j) for i in keys(mdp.trans) for j in mdp.trans[i] ]
        #println(state_pairs)
        true_values = DataFrame( run = [run for i in 1:num_rewards],
                                 c = [i for i in 1:num_rewards],
                                 theta = th_c,
                                 std = [r_std  for i in 1:num_rewards])

        # println("true mdp: $true_values")
        #priors = DataFrame(theta = [th_c[i] for i in 1:num_chains],
        priors = DataFrame(theta = [0 for i in 1:num_rewards],
                           std = [(sqrt(r_std^2) + i) for i in 1:num_rewards],
                           c = [i for i in 1:num_rewards])
        # println("priors: $priors")
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
                   num_chains=1,
                   epochs=epochs,
                   steps=H,
                   is_thompson_sampling=is_thompson_sampling,
                   rng=rng) #,
                   #rev_action_map=rev_action_map)
        #print(Q_tables)
        println("(e,i,t,st,r)")
        hist
        for (e, ag, t, st, sp, a, r)  in hist
          println("pushing $((e, ag, t, st, sp, a, r))")
          push!(runs, (run, e, ag, t, st, sp, a, r))
        end
        append!(true_values_list, true_values)

    end
    return (true_values_list, runs)
end


function get_average_regret(results,  theta, num_states)
    # 1, 1, 1, 1, 1, 2
    #println("results: $results")
    #println("theta: $theta")

    li = LinearIndices((num_states, num_states))
    df = DataFrame(run = [x[1] for x in results], epoch = [x[2] for x in results], agent=[x[3] for x in results],
               time = [x[4] for x in results], c = [li[x[5],x[6]] for x in results],  a = [x[7] for x in results],
               reward = [x[8] for x in results])
    df_R = df[df.reward.!==0,:]
    th_max = by(theta, :run, :theta => maximum)
    th_min = by(theta, :run, :theta => minimum)
    theta_joined = join(join(theta, th_max, on = :run), th_min, on = :run)
    df_joined = join(df_R, theta_joined, on = [:run,:c] )
    # needs to be max theta
    df_joined.Regret = df_joined.theta_maximum - df_joined.reward
    df_joined
end

function get_regret(results,  theta, num_states)
    # 1, 1, 1, 1, 1, 2
    #println("results: $results")
    #println("theta: $theta")

    li = LinearIndices((num_states, num_states))
    #print("li:$li")
    #print(theta)
    df = DataFrame(run = [x[1] for x in results], epoch = [x[2] for x in results], agent=[x[3] for x in results],
               time = [x[4] for x in results], s1=[x[5] for x in results], s2=[x[6] for x in results], c = [li[x[5],x[6]] for x in results],
               action = [x[7] for x in results],
               reward = [x[8] for x in results])
    #println(df[df.c.==1,:])
    #println(theta[theta.c.==1,:])
    # df_R = df[df.reward.!==0,:]
    #th_max = by(theta, :run, :theta => maximum)
    #th_min = by(theta, :run, :theta => minimum)
    #theta_joined = join(join(theta, th_max, on = :run), th_min, on = :run)
    theta.theta=[Float64(atheta) for atheta in theta.theta]
    df_joined = join(df, theta, on = [:run,:c] )
    #print(df_joined[df_joined.c.==1,:])
    # needs to be max theta
    df_joined.Regret = df_joined.theta .- df_joined.reward
    df_joined
end


# Full loop
function run_all(rng)
    full_df = nothing;
    base = 100

    for num_agents in [1000]
        println("*****************number of agents:$num_agents*******************")
        num_chains = 10
        num_states = 100
        epochs = 1
        nruns = 1
        H = 15
        do_update_priors() = true
        for run_type in ["UCB"] #"Thompson Sampling", "Seed Sampling", "UCB"]
            println("------------------Run Type:$run_type------------------------")
            if run_type == "Thompson Sampling"
                true_values_list, runs = do_runs(nruns, epochs, num_agents, num_states, H,
                                                 do_update_priors, ucb_update_priors,  ssgn_mdp_builder, true,
                                                 base, rng)
            elseif run_type == "UCB"
                true_values_list, runs = do_runs(nruns, epochs, num_agents, num_states, H,
                                                 do_update_priors, ucb_update_priors, ucb_mdp_builder, false,
                                                 base, rng)
            else
                true_values_list, runs = do_runs(nruns, epochs, num_agents, num_states, H,
                                                 do_update_priors, ucb_update_priors,  ssgn_mdp_builder, false,
                                                 base, rng)

            end

            chain_len = num_states
            num_chains = num_chains
            regret = get_regret(runs, true_values_list, num_states)
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
    #full_df[full_df[:Regret] .< 0.0,:Regret] = 0.0
    CSV.write("mr_simulation_2.csv", full_df)
    full_df
end

rng = MersenneTwister()
