import json
import math
import matplotlib.pyplot as plt

processed_data = open('segmented_data.json')
samples = json.load(processed_data)
processed_data.close()


def linfit(x_vals, y_vals):
    n = len(x_vals)
    sx = sum(x_vals)
    sx2 = sum([x * x for x in x_vals])
    sy = sum(y_vals)
    sxy = sum([x * y for (x, y) in zip(x_vals, y_vals)])
    m = (n*sxy - sx*sy)/(n*sx2 - sx*sx)
    b = (sy - m*sx)/n

    y_avg = sy / n
    ss_res = sum((y - (m*x + b))**2 for (x,y) in zip(x_vals, y_vals))
    ss_tot = sum((y - y_avg)**2 for y in y_vals)
    r2 = 1 - ss_res/ss_tot
    return m, b, r2


def opt_linfit(x_vals, y_vals):
    n = len(x_vals)
    n_quarter = n//4
    fits = dict()
    for start in range(n_quarter):
        for stop in range(n-n_quarter, n+1):
            fits[(start, stop)] = linfit(x_vals[start:stop], y_vals[start:stop])
    max_r2 = (0, (0, 0, 0))
    for (key, value) in fits.items():
        if value[2] > max_r2[1][2]:
            max_r2 = key, value
    return max_r2


u_r2_avg, d_r2_avg, u_r2_opt_avg, d_r2_opt_avg = 0,0,0,0
u_min_idx, d_min_idx = -1, -1
u_min_r2, d_min_r2 = 1, 1
for (number, sample) in samples.items():
    data = sample["data"]
    u_range = sample["upslope_range"]
    d_range = sample["downslope_range"]
    data_u = data[u_range[0]:u_range[1]]
    data_d = data[d_range[0]:d_range[1]]

    u_lin = linfit(range(len(data_u)), data_u)
    d_lin = linfit(range(len(data_d)), data_d)

    u_opt_lin = opt_linfit(range(len(data_u)), data_u)
    d_opt_lin = opt_linfit(range(len(data_d)), data_d)

    if u_opt_lin[1][2] < u_min_r2:
        u_min_r2 = u_opt_lin[1][2]
        u_min_idx = number
    if d_opt_lin[1][2] < d_min_r2:
        d_min_r2 = d_opt_lin[1][2]
        d_min_idx = number

    u_r2_avg += u_lin[2]
    d_r2_avg += d_lin[2]
    u_r2_opt_avg += u_opt_lin[1][2]
    d_r2_opt_avg += d_opt_lin[1][2]

    v_t = u_opt_lin[1][0] * (1/520 * 10**-3)
    v_2 = -d_opt_lin[1][0]
    V_stop = sample["stp_volt"]

    C_1 = 6 * math.pi * (1*10**-3)*(1.827*10**-5)*(6*10**-3)

    Q1 = C_1 * (v_t ** 1.5) / V_stop
    Q2 = (v_t + v_2) * (v_t ** 0.5) / 699
    print(round(Q1 / (1.602*10**-19)/1.02, 2))

u_r2_avg /= len(samples)
d_r2_avg /= len(samples)
u_r2_opt_avg /= len(samples)
d_r2_opt_avg /= len(samples)

print("AVG R2 (UP):\n%f" % u_r2_avg)
print("AVG R2 (UP - OPT):\n%f" % u_r2_opt_avg)
print("AVG R2 (DOWN):\n%f" % d_r2_avg)
print("AVG R2 (DOWN - OPT):\n%f" % d_r2_opt_avg)
print(u_min_idx, u_min_r2)
print(d_min_idx, d_min_r2)

plt.subplot(1,2,1)
plt.plot(samples[u_min_idx]["data"])
plt.subplot(1,2,2)
plt.plot(samples[d_min_idx]["data"])
plt.show()