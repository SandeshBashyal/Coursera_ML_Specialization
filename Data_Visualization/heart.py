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
#barplot
plt.figure(figsize=(10,6))
plt.title("Heart Attack Analysis and Prediction Data Set")
sns.barplot(x=heart_data.sex, y=heart_data['cp'])
plt.ylabel("CP")
plt.xlabel("SEX")
plt.show()
#code is same as heart.ipynb, just add plt.show()