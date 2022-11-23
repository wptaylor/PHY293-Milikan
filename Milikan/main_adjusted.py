import json
import math

import matplotlib.pyplot as plt
import better_regression_optimizer as reg

p_o = 875.3
p_a = 1.204
g = 9.81
eta = 1.827 * 10 ** -5
d = 6 * 10 ** -3

C = 9 * math.pi * d * (2 * eta ** 3 / (g * (p_o - p_a))) ** (1 / 2)

fps = 5
m_per_px = 1 / (540 * 10 ** 3)

segmented_data = open('segmented_data_fixed.json')
samples = json.load(segmented_data)
segmented_data.close()

up_last, down_last = [1, ()], [1, ()]

Q1, Q2 = [], []
for (number, sample) in samples.items():
    data = sample["data"]
    u_range = sample["upslope_range"]
    d_range = sample["downslope_range"]
    data_u = data[u_range[0]:u_range[1]]
    data_d = data[d_range[0]:d_range[1]]

    u_opt_lin = reg.opt_linfit(list(range(len(data_u))), data_u)
    d_opt_lin = reg.opt_linfit(list(range(len(data_d))), data_d)

    if u_opt_lin[2] < up_last[0]:
        up_last[0] = u_opt_lin[2]
        up_last[1] = number
    if d_opt_lin[2] < down_last[0]:
        down_last[0] = d_opt_lin[2]
        down_last[1] = number

    v_t = u_opt_lin[1][0] * fps * m_per_px
    v_2 = -d_opt_lin[1][0] * fps * m_per_px
    V_stop = sample["stp_volt"]
    V_up = 699

    q1 = C * (v_t ** 1.5) / V_stop
    q2 = C * (v_t + v_2) * (v_t ** 0.5) / V_up

    Q1.append(q1)
    Q2.append(q2)

print(down_last, up_last)

samples.pop(down_last[1])
samples.pop(up_last[1])

print(C)
Q1.sort()
Q2.sort()

segments = [x * 10 ** -19 for x in [2.6, 4.3, 6, 7.6, 9, 10, 13]]
groups = [[]]
i = 0
for q in Q2:
    if i >= len(segments):
        groups.pop()
        break
    if q < segments[i]:
        groups[i].append(q)
    else:
        i += 1
        groups.append([])

print(groups)
group_avgs = [sum(g) / len(g) for g in groups if len(g) > 0]
print(group_avgs)
charge = []
for j in range(len(group_avgs)):
    charge.append(group_avgs[j] / (j + 1))
print(charge)
print(sum(charge) / len(charge))

segmented_data = open('segmented_data_fixed.json', 'w')
json.dump(samples, segmented_data, indent=4)
segmented_data.close()


plt.figure(1)
plt.scatter(Q1, Q2)
plt.xlabel("Q1: Method 1")
plt.ylabel("Q2: Method 2")
plt.show()

plt.figure(2)
plt.scatter(range(len(Q2)), Q2)
plt.ylabel("Q2: Method 2")
plt.show()
