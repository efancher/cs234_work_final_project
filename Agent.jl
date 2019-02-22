module PFAgent
    using POMDPPolicies
    using POMDPSimulators
    curry(f, x) = (xs...) -> f(x, xs...)
    function run_chain!(;policies, found_target, true_mdp, update_Q, n_agents, n_states,
                        Q_tables, N_tables, epochs, steps, rev_action_map, stop_early)
        for e in 1:epochs
            agents = []
            for i in 1:n_agents
                push!(agents,
                      Iterators.Stateful(stepthrough(true_mdp, FunctionPolicy(policies[i]), "s,a,r,sp,t", max_steps=steps)))
            end

            #println("epoch: $e")
            done = false
            t = 0
            start = 1
            while ! done
               done = true
               for i in 1:n_agents
                    if i > start
                        start += 1
                        done = false
                        break
                    end
                    if isempty(agents[i])
                        println("agent $i is done")
                        continue
                    end
                    res = popfirst!(agents[i])
                    r = res[:r]
                    N_tables[i][res[:s],rev_action_map[res[:a]]] += 1
                    update_Q(Q_tables[i],res..., rev_action_map)
                    t = res[:t]
                    if e % 10 == 0 || stop_early
                      println("e: $e, t: $t, print agent $i result: $res")
                    end
                    if found_target(r) && stop_early
                        return "Done!"
                    end
               end
               for i in 1:n_agents
                    if ! isempty(agents[i])
                        done = false
                    end
               end
            end
        end
    end
    function setup_agents(states, num_states, num_agents, actions, num_actions, policy_function, AGMDP)
        Q_tables = []
        policies = []
        mdps = Any[]
        empty_N = zeros( num_states+2, num_actions)
        N_tables = Dict{Int32, typeof(empty_N)}()
        Central_Q_table = Dict{Int32, Dict{Int32, Float32}}()

        for i in 1:num_agents
            push!(Q_tables, Central_Q_table)
            for state in states
              Q_tables[i][state] = Dict{Int32, Float32}()
              # print(Q_tables[i])
              for action in actions
                 Q_tables[i][state][action] = 0
              end
            end
            N_tables[i] = deepcopy(empty_N)
            push!(policies, curry(curry(curry(curry(policy_function, Q_tables), N_tables),i), actions))
            push!(mdps, AGMDP)
        end
        return (Q_tables, N_tables, policies, mdps)
    end
end
