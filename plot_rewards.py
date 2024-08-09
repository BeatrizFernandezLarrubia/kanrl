import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample DataFrame creation (replace this with your actual data)
df = pd.read_csv("final_test/hr1_v4.csv")
x_column = "num_legs"
print(df)

# Set the plot style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(10, 6))

# Loop through each method and plot
methods = df['method'].unique()
colors = sns.color_palette("husl", len(methods))  # Generate distinct colors for each method

for i, method in enumerate(methods):
    method_data = df[df['method'] == method].sort_values([x_column])
    plt.plot(method_data[x_column], method_data['mean_reward'], "-o", label=f'{method}', color=colors[i])
    plt.fill_between(method_data[x_column], 
                     method_data['mean_reward'] - method_data['std_reward'], 
                     method_data['mean_reward'] + method_data['std_reward'], 
                     color=colors[i], alpha=0.3)

# Add titles and labels
plt.title('Mean Reward vs Number of Ant legs (noise std=0.2)')
plt.xlabel('Number of Ant legs')
plt.ylabel('Reward')
plt.xticks(sorted(df[x_column].unique()))
plt.legend()
plt.show()

# Add titles and labels
# plt.title('Mean Reward vs Standard Deviation of Noise on actions (4-legged Ant)')
# plt.xlabel('Standard Deviation of Noise')
# plt.ylabel('Reward')
# plt.xticks(sorted(df[x_column].unique()))
# plt.legend()
# plt.show()
