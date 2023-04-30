import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# pd.__version__
# pd.plotting.register_matplotlib_converters()
# a=pd.DataFrame({'a': [1, 2]})
# print(a)
myfile='heart.csv'
heart_data=pd.read_csv(myfile)
print(heart_data)