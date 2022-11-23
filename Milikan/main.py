import json
import math

import matplotlib.pyplot as plt
import numpy as np
import better_regression_optimizer as reg

p_o = 875.3
p_a = 1.204
g = 9.81
eta = 1.827 * 10 ** -5
d = 6 * 10 ** -3

C = 9 * math.pi * d * (2 * eta ** 3 / (g * (p_o - p_a))) ** (1 / 2)

fps = 5
m_per_px = 1 / (540 * 10 ** 3)

segmented_data = open('segmented_data.json')
samples = json.load(segmented_data)
segmented_data.close()

R2u, R2d = 0, 0
Q1, Q2 = [], []
rs = []
i = -1
fig = plt.figure()
fig.subplots_adjust(0.075, 0.075, 1 - 0.075, 1 - 0.075, 0.6, 0.45)
for (number, sample) in samples.items():
    i += 2
    data = sample["data"]
    u_range = sample["upslope_range"]
    d_range = sample["downslope_range"]
    data_u = data[u_range[0]:u_range[1]]
    data_d = data[d_range[0]:d_range[1]]

    u_opt_lin = reg.opt_linfit(list(range(len(data_u))), data_u)
    d_opt_lin = reg.opt_linfit(list(range(len(data_d))), data_d)

    R2u += u_opt_lin[2]
    R2d += d_opt_lin[2]

    plt.subplot(8, 15, i)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.plot(data_u[u_opt_lin[0][0]:u_opt_lin[0][1] + 1])
    plt.subplot(8, 15, i + 1)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.plot(data_d[d_opt_lin[0][0]:d_opt_lin[0][1] + 1])

    v_t = u_opt_lin[1][0] * fps * m_per_px
    v_2 = -d_opt_lin[1][0] * fps * m_per_px
    V_stop = sample["stp_volt"]
    V_up = 699

    q1 = C * (v_t ** 1.5) / V_stop
    q2 = C * (v_t + v_2) * (v_t ** 0.5) / V_up
    r = 3 * (eta * v_t / (2 * g * (p_o - p_a))) ** (1 / 2)
    rs.append(r)
    Q1.append(q1)
    Q2.append(q2)

plt.show()


Q2.sort()
segments = [x * 10 ** -19 for x in [2.6, 4.3, 6, 7.6, 9, 10, 13]]
groups = [[]]
i, j = 0, 0
while i < len(segments):
    if Q2[j] < segments[i]:
        groups[i].append(Q2[j])
        j += 1
    else:
        i += 1
        groups.append([])

group_avgs = [sum(g) / len(g) for g in groups if len(g) > 0]
group_stds = [np.std(g) for g in groups if len(g) > 0]
charge = []
for j in range(len(group_avgs)):
    charge.append(group_avgs[j] / (j + 1))

print("Const:", C)
print("Avg R2:", R2u / 60, R2d / 60)
print("Std Dev a:", sum(group_stds) / len(group_stds))
print("Std Dev b1:", np.std(charge))
print("Charge: ", sum(charge) / len(charge))
print("% Error:", (10 ** 19 * sum(charge) / len(charge) - 1.602) / 1.602 * 100)
print("R: ", sum(rs) / len(rs))
print("R stddev: ", np.std(rs))

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
plt.scatter(Q1, Q2)
plt.xlabel("Q1: Method 1")
plt.ylabel("Q2: Method 2")
plt.show()

plt.figure(2)
plt.scatter(range(len(Q2)), Q2)
plt.ylabel("Charge (Coulombs)")
plt.xlabel("Sample  ")
plt.show()
