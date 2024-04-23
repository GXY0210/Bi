import matplotlib.pyplot as plt
import numpy as np

X = ['horse', 'blue+whale', 'sheep', 'seal', 'bat', 'giraffe', 'rat', 'bobcat', 'walrus', 'dolphin']

real_prior = [0.14441513, 0.04591029, 0.11099384, 0.08953386, 0.0705365, 0.19349164, 0.05329815, 0.11275286, 0.0592788,
              0.11978892]
estimate_prior_a = [0.22145998, 0.1062445, 0.035708, 0.03236588, 0.07141601, 0.20175901,
                    0.04538259, 0.10976253, 0.11398417, 0.06191733]

estimate_prior_v = [0.13421284, 0.11873351, 0.11662269, 0.05030783, 0.10483729, 0.18364116,
                    0.05382586, 0.09973615, 0.08671944, 0.05136324]

estimate_prior_h = [0.10343008, 0.05136324, 0.10272647, 0.06983289, 0.06033421, 0.17203166,
                    0.17537379, 0.0826737, 0.07405453, 0.10817942]

estimate_prior_vha = [0.13104661, 0.06174142, 0.1298153, 0.09533861, 0.07370273, 0.18962181,
                      0.05470536, 0.10343008, 0.05241865, 0.10817942]

fig, axs = plt.subplots(4, 1, figsize=(10, 12))

X_axis = np.arange(len(X))
width = 0.25

axs[0].bar(X_axis - width / 2, real_prior, width=width, label='Real prior', color='orange')
axs[0].bar(X_axis + width / 2, estimate_prior_a, width=width, label='Estimated prior by att', color='gray')
axs[0].set_xticks(X_axis)
axs[0].set_xticklabels(X)
axs[0].set_title('Prior Probability (Att)')
axs[0].set_xlabel('Test classes')
axs[0].set_ylabel('Probability')
axs[0].legend()

axs[1].bar(X_axis - width / 2, real_prior, width=width, label='Real prior', color='orange')
axs[1].bar(X_axis + width / 2, estimate_prior_v, width=width, label='Estimated prior by v', color='blue')
axs[1].set_xticks(X_axis)
axs[1].set_xticklabels(X)
axs[1].set_title('Prior Probability (V)')
axs[1].set_xlabel('Test classes')
axs[1].set_ylabel('Probability')
axs[1].legend()

axs[2].bar(X_axis - width / 2, real_prior, width=width, label='Real prior', color='orange')
axs[2].bar(X_axis + width / 2, estimate_prior_h, width=width, label='Estimated prior by h', color='green')
axs[2].set_xticks(X_axis)
axs[2].set_xticklabels(X)
axs[2].set_title('Prior Probability (H)')
axs[2].set_xlabel('Test classes')
axs[2].set_ylabel('Probability')
axs[2].legend()

axs[3].bar(X_axis - width / 2, real_prior, width=width, label='Real prior', color='orange')
axs[3].bar(X_axis + width / 2, estimate_prior_vha, width=width, label='Estimated prior by vha', color='yellow')
axs[3].set_xticks(X_axis)
axs[3].set_xticklabels(X)
axs[3].set_title('Prior Probability (VHA)')
axs[3].set_xlabel('Test classes')
axs[3].set_ylabel('Probability')
axs[3].legend()

plt.tight_layout()
plt.show()
