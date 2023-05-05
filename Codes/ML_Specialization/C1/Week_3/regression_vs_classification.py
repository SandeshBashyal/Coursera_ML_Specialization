import numpy as np
# %matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import dlc, plot_data
from plt_one_addpt_onclick import plt_one_addpt_onclick

x_train1 = np.array([0.,1,2,3,4,5])
y_train1 = np.array([0, 0, 0, 1, 1, 1])
x_train2 = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train2 = np.array([0, 0, 0, 1, 1, 1])

pos = y_train1 == 1 #pos 1=True, 0=False
neg = y_train1 == 0
print(pos)
print(neg)

fig,ax = plt.subplots(1, 2, figsize=(8,3))
#plot 1, single variable
ax[0].scatter(x_train1[pos],y_train1[pos],marker='x', s=80, c='red', label='y=1')
ax[0].scatter(x_train1[neg], y_train1[neg], marker='o', s = 100, label='y=0', facecolors = 'none',
              edgecolors=dlc["dlblue"], lw=3)
ax[0].set_ylim(-0.08,1.1)
ax[0].set_ylabel('y1', fontsize=12)
ax[0].set_xlabel('x1', fontsize=12)
ax[0].set_title('one variable plot')
ax[0].legend()

#plot 2, two variables
plot_data(x_train2, y_train2, ax[1])
ax[1].axis([0, 4, 0, 4])
ax[1].set_ylabel('$y2$', fontsize=12)
ax[1].set_xlabel('$x2$', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()
plt.tight_layout()
plt.show()

w_in = np.zeros((1))
b_in = 0
plt.close('all')
addpt = plt_one_addpt_onclick(x_train1, y_train1, w_in, b_in, logistic=False)
plt.show() #in jupyter no need of plt.show