library(ggplot2)
#results <- read.csv("~/src/cs234/cs234_work/final_project/pc_simulation.csv")
#results1 <- read.csv("~/src/cs234/cs234_work/final_project/pc_simulation_upto1000.csv")
#results2 <- read.csv("~/src/cs234/cs234_work/final_project/pc_simulation_10000.csv")
#results <- rbind(results1, results2)
results_mr <- read.csv("~/src/cs234/cs234_work/final_project/mr_simulation_2_runs_46_46.csv")
library(ggplot2)
# ln = ggplot(data=results, aes(x=agent, y=Regret, color=name)) + 
#   geom_line() + ggtitle("Average Regret per Agent Count by Algorithm Type") + 
#   theme(plot.title = element_text(hjust = 0.5))
# #jpeg("~/src/cs234/cs234_work/final_project/project_update/results.jpg")
# plot(ln)
#dev.off()
library(dplyr)
sqrt()
unique(results_mr$run)
rs = seq(1,nrow(tmpd_df))
rs
plot(tmpd_df$agent, tmpd_df$Regret)
ln = ggplot(data=results_mr, aes(x=agent, y=reward)) + 
  geom_line() + facet_grid(~name) + # ggtitle("Average Regret per Agent Count by Algorithm Type") + 
  theme(plot.title = element_text(hjust = 0.5))
ln

ln = ggplot(data=results_mr, aes(x=agent, y=Regret)) + 
  geom_line() + facet_grid(~name) + # ggtitle("Average Regret per Agent Count by Algorithm Type") + 
  theme(plot.title = element_text(hjust = 0.5)) + scale_y_log10()
ln
cum_regret_by_algo = mutate(group_by(results_mr,run, overall_step), cumsum=cumsum(Regret), reward=reward)
# first need to get average regret, how? per agent?

cum_regret_by_algo2 = group_by(results_mr, name, agent) %>% summarise(sum_agent_regret=mean(Regret)) %>% 
                                                                   mutate(cumsum_regret = cumsum(sum_agent_regret))
head(cum_regret_by_algo2)

ln = ggplot(data=cum_regret_by_algo2, aes(x=agent, y=cumsum_regret, color=name)) + 
     geom_line() + ggtitle("Average Regret per Agent Count by Algorithm Type") + 
     theme(plot.title = element_text(hjust = 0.5))
ln
cum_regret_by_algo2 = group_by(results_mr, agent, name) %>% summarise(agent_regret=mean(Regret)) %>% 
  mutate(cumsum_regret = cumsum(agent_regret))

ln = ggplot(data=cum_regret_by_algo2, aes(x=agent, y=cumsum_regret, color=name)) + 
  geom_line() + ggtitle("Average Regret per Agent Count by Algorithm Type") + 
  theme(plot.title = element_text(hjust = 0.5))