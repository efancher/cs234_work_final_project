results <- read.csv("~/src/cs234/cs234_work/final_project/project_update/results.csv")

library(ggplot2)
ln = ggplot(data=results, aes(x=agents, y=averageregret, color=algorithm)) + 
  geom_line() + ggtitle("Average Regret per Agent Count by Algorithm Type") + 
  theme(plot.title = element_text(hjust = 0.5))
#jpeg("~/src/cs234/cs234_work/final_project/project_update/results.jpg")
plot(ln)
#dev.off()
