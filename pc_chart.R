library(ggplot2)
#results <- read.csv("~/src/cs234/cs234_work/final_project/pc_simulation.csv")
#results1 <- read.csv("~/src/cs234/cs234_work/final_project/pc_simulation_upto1000.csv")
#results2 <- read.csv("~/src/cs234/cs234_work/final_project/pc_simulation_10000.csv")
#results <- rbind(results1, results2)
results <- read.csv("~/src/cs234/cs234_work/final_project/pc_simulation_rerun2.csv")
# library(ggplot2)
# ln = ggplot(data=results, aes(x=agent, y=Regret, color=name)) + 
#   geom_line() + ggtitle("Average Regret per Agent Count by Algorithm Type") + 
#   theme(plot.title = element_text(hjust = 0.5))
# #jpeg("~/src/cs234/cs234_work/final_project/project_update/results.jpg")
# plot(ln)
#dev.off()
library(dplyr)

filt = results[results$epoch==1,] %>%
  group_by(name, num_agents) %>%
  summarise(
    
    n = max(agent),
    AverageRegret = mean(Regret, na.rm = TRUE)
  )
print(filt)
# write.csv(filt, file="~/src/cs234/cs234_work/final_projectpc_summary.csv")

# results <- read.csv("~/src/cs234/cs234_work/final_project/project_update/results.csv")
filt$algorithm = filt$name
library(ggplot2)
ln = ggplot(data=filt, aes(x=num_agents, y=AverageRegret, color=algorithm)) + 
  geom_line() + ggtitle("Average Regret per Agent Count by Algorithm Type") + 
  theme(plot.title = element_text(hjust = 0.5)) + scale_x_log10()
#jpeg("~/src/cs234/cs234_work/final_project/project_update/results.jpg")
plot(ln)
#dev.off()

