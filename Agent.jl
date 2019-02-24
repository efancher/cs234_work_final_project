module PFAgent
    using POMDPs
    using POMDPPolicies
    using POMDPSimulators
    curry(f, x) = (xs...) -> f(x, xs...)
    function run_chain!(;policies, found_target, mdps, update_Q, n_agents, n_states,
                        Q_tables, N_tables, epochs, steps, rev_action_map, stop_early)
        r_history =[]
        for e in 1:epochs
            agents = []
            for i in 1:n_agents
                push!(agents,
                      Iterators.Stateful(stepthrough(mdps[i], FunctionPolicy(policies[i]), "s,a,r,sp,t",
                      max_steps=steps)))
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
                    t = res[:t]
                    st = res[:s]
                    push!(r_history, (e,i,t,st,r))
                    # println("before: i:$i, s:$(res[:s]), a:$(rev_action_map[res[:a]]), N:$(N_tables[i][res[:s],rev_action_map[res[:a]]])")
                    N_tables[i][res[:s],rev_action_map[res[:a]]] += 1
                    # println("after: i:$i, s:$(res[:s]), a:$(rev_action_map[res[:a]]), N:$(N_tables[i][res[:s],rev_action_map[res[:a]]])")

                    update_Q(Q_tables[i],res..., rev_action_map)
                    if e % (floor(epochs/10)) == 0 || stop_early
                       println("e: $e, t: $t, agent $i, result: $res")
                    end
                    if found_target(r) && stop_early
                        return  r_history  # Note after this all actions will be optimal, so we don't need to
                                           # record them
                    end
               end
               for i in 1:n_agents
                    if ! isempty(agents[i])
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
            push!(Q_tables, Central_Q_table)
            for state in states
              Q_tables[i][state] = Dict{Int32, Float32}()
              # print(Q_tables[i])
              for action in actions
                 Q_tables[i][state][action] = 0.0
              end
            end
            N_tables[i] = deepcopy(empty_N)
            push!(policies, curry(curry(curry(curry(policy_function, Q_tables), N_tables),i), actions))
        end
        return (Q_tables, N_tables, policies)
    end
end
