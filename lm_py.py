#!/usr/bin/env python
# coding: utf-8

# # Jupyter Notebook First Assignment Python Version

# ## Elijah Blige

# ## Reading in data

# In[3]:


import pandas as pd
import sys

if len(sys.argv) > 1:
    datafile = sys.argv[1]
   
else:
    print("Please input a valid Filename")
    sys.exit(-1)
data = pd.read_csv(sys.argv[1])

# ## Creating the scatterplot

# In[4]:


import matplotlib.pyplot as plt
x = data['x']
y = data['y']
plt.scatter(x,y)
plt.xlabel('X')
plt.ylabel('Y')           
plt.show()
plt.savefig("py_orig.png")

# ## Model the data (linear model)

# In[5]:


import numpy as np
arrayX = np.array(x).reshape((-1,1))
arrayY = np.array(y)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(arrayX,arrayY)
y_pred = model.predict(arrayX)
r_sq = model.score(arrayX,arrayY)


# # Ploting model with the original data

# In[6]:


plt.scatter(x,y)
plt.plot(x,y_pred)
plt.xlabel('X')
plt.ylabel('Y')  
plt.show()
plt.savefig("py_lm.png")

# In[ ]:




