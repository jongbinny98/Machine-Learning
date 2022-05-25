# importing the required module 
import matplotlib.pyplot as plt 
    
# x axis values remaining training data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30] 

# corresponding y axis values Accuracy
y = [0.75, 0.625, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.625, 0.625, 0.75, 0.25, 0.25, 0.125] 
    
# plotting the points  
plt.plot(x, y) 

# plotting the title
plt.title("K-th Nearest Neighbor")

# naming the x axis 
plt.xlabel('K') 
# naming the y axis 
plt.ylabel('Accuracy') 

# function to show the plot 
plt.show() 