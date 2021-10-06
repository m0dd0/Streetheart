import matplotlib.pyplot as plt
import time

plt.ion()

fig, ax = plt.subplots()
ax.plot(list(range(5)), list(range(5)))
(line,) = ax.plot([0, 0], [0, 0])

for i in range(5):
    line.set_xdata([i, i + 1])
    line.set_ydata([i, i + 1])
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.5)
    # fig.show()
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # time.sleep(3)
    print(i)
# plt.show()