using POMDPs
using Random # for AbstractRNG
using POMDPModelTools
using Pkg
using Dates

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

function build_random_path(g_len, sigma)
  trans = Dict{Int64, Array{Int64}}()
  for i in 1:g_len

        if !(i  in keys(trans))
          trans[i] = []
        end
        for j in 1:g_len
          
          p = 2 * log(g_len)/g_len

          if rand(Bernoulli(p)) == 1
                append!(trans[i], j)
          end
        end
  end


  is = keys(trans)
  Rs = zeros(Float64,g_len,g_len)
  for i in is
    for j in trans[i]

      Rs[i,j] = exp(randn(Float64)*4)
    end
  end
  new_mdp = PMaxRewardPathMDP(g_len, .95,
        Rs,
        trans, sigma)

  return(new_mdp)
 end


function ucb_pol(li, policy, priors, i, actions, s)
    best_action = action(policy, s)

    return action(policy, s)
end


curry(f, x) = (xs...) -> f(x, xs...)
function ucb_mdp_builder(rng, true_mdp, priors, i, num_states, num_chains, steps, start_state)
  # build mdp

  mdp = deepcopy(true_mdp)


  li = LinearIndices((num_states, num_states))
  mdp.Rs = reshape([prior.theta + 6* prior.std for prior in eachrow(priors)],num_states,num_states)

  solver = SparseValueIterationSolver(max_iterations=10000000, belres=1e-6, include_Q=true)#, verbose=true) # initializes the Solver type
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

  mdp = deepcopy(true_mdp)

  li = LinearIndices((num_states, num_states))


  mdp.Rs = reshape([(randn(rng, Float32) * prior.std + prior.theta)  for prior in eachrow(priors)],num_states,num_states)

  solver = SparseValueIterationSolver(max_iterations=10000000, belres=1e-6, include_Q=true)# initializes the Solver type
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


function ucb_update_priors(priors, hist, true_vals, true_mdp, num_states)

    pc = deepcopy(priors)
    hist_last_row = hist[length(hist)]
    last_row = (epoch=hist_last_row[1], agent=hist_last_row[2],
               time=hist_last_row[3], state=hist_last_row[4], state2=hist_last_row[5], a=hist_last_row[6],
               reward=hist_last_row[7])


    new_theta = []
    new_std = []
    new_c = []
    cs = CartesianIndices(size(true_mdp.Rs))
    li = LinearIndices((num_states, num_states))
    c = li[last_row.state, last_row.state2]

            mu_p = priors[priors.c.==c,:].theta[1]

            true_std = true_mdp.sigma
            t_s_2 = true_std^2

            s_p_2 = (priors[priors.c.==c,:].std[1])^2

            tmp_theta = (t_s_2 * mu_p + s_p_2 *(log(last_row.reward) + t_s_2/2))/(s_p_2+t_s_2)
            tmp_std = sqrt(s_p_2*t_s_2/(s_p_2+t_s_2))

        pc[pc.c.==c,:theta] = tmp_theta
        pc[pc.c.==c,:std] = tmp_std

    tmp_joined = join(priors, pc, on = :c, makeunique = true)


    return pc
end

using Distributions
function do_runs(start_run, end_run, epochs, num_agents,  num_states, H, update_priors_check, update_priors,
                 mdp_iter_builder, is_thompson_sampling, base, rng)

    r_mean = 0
    r_std = 1

    runs = []
    true_values_list = DataFrame( run = [],
                                 c = [],
                                 theta = [],
                                 std = [])
    for run in start_run:end_run

        run_start = Dates.value(Dates.now())
        println("run: $run")
        sigma = sqrt(.01)
        mdp = build_random_path(num_states, sigma)
        solver = SparseValueIterationSolver(max_iterations=10000000, belres=1e-6, include_Q=true)# initializes the Solver type
        vip = solve(solver, mdp)
        li = LinearIndices((num_states, num_states))
        cs = CartesianIndices(size(mdp.Rs))

        th_c = reshape(mdp.Rs, 1,:)

        th_c_sd = sqrt(var(th_c))
        r_std = th_c_sd
        num_rewards = length(th_c)
        th_c = dropdims(th_c;dims=1)
        state_pairs = [(i,j) for i in keys(mdp.trans) for j in mdp.trans[i] ]

        true_values = DataFrame( run = [run for i in 1:num_rewards],
                                 c = [i for i in 1:num_rewards],
                                 theta = th_c,
                                 std = [r_std  for i in 1:num_rewards])


        priors = DataFrame(theta = [0.0 for i in 1:num_rewards],
                           std = [(sqrt(r_std^2) + i) for i in 1:num_rewards],
                           c = [i for i in 1:num_rewards])

        do_update_priors() =  false

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
                   rng=rng)


        run_end= Dates.value(Dates.now())
        #hist
        for (e, ag, t, st, sp, a, r, overall_step)  in hist

          best_act = action(vip, Vec2(1,st))

          best_reward_un_mod = maximum([reward(mdp, st, act, sp) for act in actions(mdp, Vec2(1,st))])
          expected_reward_un_mod = reward(mdp, st, a, sp)

          best_reward = exp(log(best_reward_un_mod) - ((mdp.sigma)^2)/2)
          expected_reward = exp(log(expected_reward_un_mod) - ((mdp.sigma)^2)/2)

          push!(runs, (run, e, ag, t, st, sp, a, r, overall_step, best_reward, expected_reward, run_start, run_end))
        end
        append!(true_values_list, true_values)

    end
    return (true_values_list, runs)
end


function get_regret(results,  theta, num_states)


    li = LinearIndices((num_states, num_states))
    #print("li:$li")
    #print(theta)
    df = DataFrame(run = [x[1] for x in results], epoch = [x[2] for x in results], agent=[x[3] for x in results],
               time = [x[4] for x in results], s1=[x[5] for x in results], s2=[x[6] for x in results], c = [li[x[5],x[6]] for x in results],
               action = [x[7] for x in results],
               reward = [x[8] for x in results],
               overall_step = [x[9] for x in results],
               best_reward = [x[10] for x in results],
               expected_reward = [x[11] for x in results],
               run_start = [x[12] for x in results],
               run_end = [x[13] for x in results])


    theta.theta=[Float64(atheta) for atheta in theta.theta]
    df_joined = join(df, theta, on = [:run,:c] )

    df_joined.Regret = df_joined.best_reward .- df_joined.expected_reward
    df_joined
end


# Full loop
function run_all(rng, start_run, end_run)
    full_df = nothing;
    base = 100

    for num_agents in [2000]
        println("*****************number of agents:$num_agents*******************")
        num_chains = 10
        num_states = 100#100
        epochs = 1
        start_run = start_run
        end_run = end_run
        H = 10
        do_update_priors() = true
        for run_type in ["UCB", "Thompson Sampling", "Seed Sampling"]
            println("------------------Run Type:$run_type------------------------")
            if run_type == "Thompson Sampling"
                true_values_list, runs = do_runs(start_run, end_run, epochs, num_agents, num_states, H,
                                                 do_update_priors, ucb_update_priors,  ssgn_mdp_builder, true,
                                                 base, rng)
            elseif run_type == "UCB"
                true_values_list, runs = do_runs(start_run, end_run, epochs, num_agents, num_states, H,
                                                 do_update_priors, ucb_update_priors, ucb_mdp_builder, false,
                                                 base, rng)
            else
                true_values_list, runs = do_runs(start_run, end_run, epochs, num_agents, num_states, H,
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

    CSV.write("mr_simulation_2_runs_$(start_run)_$(end_run).csv", full_df)
    full_df
end

rng = MersenneTwister()
