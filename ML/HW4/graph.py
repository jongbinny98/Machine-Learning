# importing the required module 
import matplotlib.pyplot as plt 
    
# x axis values remaining training data
x = [1, 2, 3, 4, 5, 6] 
# error rate
y = [0.09, 0.27, 0.21, 0.34, 0.42, 0.68]

# plotting the points  
plt.plot(x, y) 

# plotting the title
plt.title("smooth algorithm")

# naming the x axis 
plt.xlabel('Iteration') 
# naming the y axis 
plt.ylabel('Error rate') 

# function to show the plot 
plt.show() 