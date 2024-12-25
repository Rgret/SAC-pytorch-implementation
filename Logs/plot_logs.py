import matplotlib.pyplot as plt
import pandas as pd

# Load the data from the .txt file
file_path = 'logs.txt'  # Update with your actual file path
data = pd.read_csv(file_path, sep='\t')

# Extract necessary columns for plotting
epochs = data['Epoch']
avg_q1_vals = data['AverageQ1Vals']
avg_q2_vals = data['AverageQ2Vals']
avg_log_pi = data['AverageLogPi']
loss_pi = data['LossPi']
loss_q = data['LossQ']

# Plotting the metrics
plt.figure(figsize=(14, 8))

# Plot Average Q1 and Q2 Values
plt.subplot(2, 2, 1)
plt.plot(epochs, avg_q1_vals, label='Average Q1 Values')
plt.plot(epochs, avg_q2_vals, label='Average Q2 Values')
plt.title('Average Q1 and Q2 Values over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Q Values')
plt.legend()

# Plot Average Log Pi
plt.subplot(2, 2, 2)
plt.plot(epochs, avg_log_pi, label='Average Log Pi')
plt.title('Average Log Pi over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Log Pi')
plt.legend()

# Plot Loss Pi and Loss Q
plt.subplot(2, 2, 3)
plt.plot(epochs, loss_pi, label='Loss Pi')
plt.title('Loss Pi over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss Pi')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, loss_q, label='Loss Q')
plt.title('Loss Q over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss Q')
plt.legend()

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
