import matplotlib.pyplot as plt
import numpy as np

data = np.load(f"../data/toy-data.npz")

datas = data["training_data"]
labels =data["training_labels"]
w=np.array([-0.4528,-0.5190])
alpha=0.1471

#画散点
plt.scatter(datas[:, 0], datas[:, 1], c=labels)

# Plot the decision boundary
x = np.linspace(-5, 5, 100)
y = -(w[0] * x + alpha) / w[1]
plt.plot(x, y, 'k')
# Plot the margins
## TODO
y1=-(w[0] * x + alpha + 1) / w[1]
y2=-(w[0] * x + alpha - 1) / w[1]
plt.plot(x, y1, 'k--')
plt.plot(x, y2, 'k--')
plt.show()