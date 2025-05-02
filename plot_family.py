import matplotlib.pyplot as plt

models = ['ResNet-18', 'ResNet-34', 'ResNet-50']
accuracy = [0.370, 0.339, 0.317]          # validation accuracy
params = [11.3, 21.4, 23.9]               # in Millions
flops = [296.296, 598.286, 668.107]       # in Millions

fig, axs = plt.subplots(2, 1, figsize=(6, 8), dpi=120)

# Params vs Accuracy
axs[0].plot(params, accuracy, 'ro-')
for i in range(len(models)):
    axs[0].text(params[i] + 0.5, accuracy[i], models[i], fontsize=8)
axs[0].set_xlabel("Number of Parameters (Millions)")
axs[0].set_ylabel("Validation Accuracy")
axs[0].grid(True)

# FLOPs vs Accuracy
axs[1].plot(flops, accuracy, 'bs-')
for i in range(len(models)):
    axs[1].text(flops[i] + 10, accuracy[i], models[i], fontsize=8)
axs[1].set_xlabel("FLOPs (Millions)")
axs[1].set_ylabel("Validation Accuracy")
axs[1].grid(True)

plt.tight_layout()
plt.show()
