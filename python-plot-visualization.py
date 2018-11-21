import matplotlib.pyplot as plt
import csv

x = []
y = []

with open('visuals.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None) #skip header
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[2]))

plt.scatter(x, y)
plt.plot([-3.66, -3.66], [0, 2.44], 'k-', lw=2)
plt.plot([+3.66, +3.66], [0, 2.44], 'k-', lw=2)
plt.plot([-3.66, +3.66], [2.44, 2.44], 'k-', lw=2)
plt.plot([-10, +10], [0, 0], 'k-', lw=2)
plt.xlim(right=5, left=-5)
plt.ylim(bottom=-3, top=3)
plt.show()