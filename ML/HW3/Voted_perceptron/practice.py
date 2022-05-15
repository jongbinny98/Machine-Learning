# importing the required module 
import matplotlib.pyplot as plt 
    
# x axis values remaining training data
x = [0.99, 0.98, 0.95, 0.90, 0.80, 0.00] 
# corresponding y axis values Accuracy
y = [0.78, 0.80, 0.88, 0.88, 0.96, 1.00] 
    
# plotting the points  
plt.plot(x, y) 
    
# naming the x axis 
plt.xlabel('Percentage remaining') 
# naming the y axis 
plt.ylabel('Accuracy') 

# function to show the plot 
plt.show() 