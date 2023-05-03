import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# m = len(x_train)
# print(f"Number of training examples is: {m}")

i = 0 # Change this to 1 to see (x^1, y^1)
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")
# plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
# plt.show()
w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")
print("Our First Prediction")

def compute_model(x,w,b):
    m=x.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=w*x[i]+b
    return f_wb

tmp_f_wb = compute_model(x_train, w, b)

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our first Prediction')
w=200
print(f"w: {w}")
print(f"b: {b}")
print("Our Second Prediction")
trueline = compute_model(x_train,w,b)   
# print(f"${cost_1200sqft} thousand dollars")
plt.plot(x_train, trueline, c='g',label='Our Second Prediction')
plt.legend()

x_i = 1.2
cost_sqft = w * x_i + b    
plt.axhline(y=cost_sqft,color='orange',linestyle='-')
plt.axvline(x=x_i,color='orange',linestyle='-')
plt.show()
print(f"${cost_sqft:.0f} thousand dollars")
