import matplotlib.pyplot as plt

gflops_arr = []
with open('./build/report.txt') as file:
    lines = file.readlines()
    for line in lines:
        (h, w, _, glflops) = line.split(',')
        gflops_arr.append(float(glflops[8:]))
x = [i for i in range(len(gflops_arr))]
plt.plot(x, gflops_arr)
plt.show()
