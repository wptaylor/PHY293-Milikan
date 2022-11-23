# Produce the linear-regression slope and intercept from relevant sums and number of points
def linfit(sx, sx2, sy, sxy, n):
    m = (n * sxy - sx * sy) / (n * sx2 - sx * sx)
    b = (sy - m * sx) / n
    return m, b


# Cumulative summation
def cumsum(values, reverse=False):
    if reverse:
        values.reverse()
    for i in range(1, len(values)):
        values[i] += values[i - 1]
    return


# Take the most linear contiguous subset via the r-squared statistical score.
# Reduce to as much as [size//4 - 1: size - size//4 + 1]
def opt_linfit(x_data, y_data):

    # Define useful indices
    n_total = len(x_data)
    lower_quarter = n_total // 4
    upper_quarter = n_total - lower_quarter
    n_mid = upper_quarter - lower_quarter

    # Preallocate and compute useful lists
    x2data = [x ** 2 for x in x_data]
    xydata = [x * y for x, y in zip(x_data, y_data)]

    # Compute the middle sums
    sx_mid = sum(x_data[lower_quarter:upper_quarter])
    sx2_mid = sum(x2data[lower_quarter:upper_quarter])
    sy_mid = sum(y_data[lower_quarter:upper_quarter])
    sxy_mid = sum(xydata[lower_quarter:upper_quarter])

    # Compute left and right cumulative sums
    sx_lefts = x_data[:lower_quarter]
    cumsum(sx_lefts, reverse=True)
    sx2_lefts = x2data[:lower_quarter]
    cumsum(sx2_lefts, reverse=True)
    sy_lefts = y_data[:lower_quarter]
    cumsum(sy_lefts, reverse=True)
    sxy_lefts = xydata[:lower_quarter]
    cumsum(sxy_lefts, reverse=True)

    # Right sums are already in the proper order
    sx_rights = x_data[upper_quarter:]
    cumsum(sx_rights)
    sx2_rights = x2data[upper_quarter:]
    cumsum(sx2_rights)
    sy_rights = y_data[upper_quarter:]
    cumsum(sy_rights)
    sxy_rights = xydata[upper_quarter:]
    cumsum(sxy_rights)

    r2_max = 0
    i_max, j_max = 0, 0
    m_max, b_max = 0, 0
    # Iterate through left and right subset extensions, each of length lower_quarter
    for i in range(lower_quarter):
        for j in range(lower_quarter):
            n = n_mid + i + j + 2

            # Full sums
            sx = sx_lefts[i] + sx_mid + sx_rights[j]
            sx2 = sx2_lefts[i] + sx2_mid + sx2_rights[j]
            sy = sy_lefts[i] + sy_mid + sy_rights[j]
            sxy = sxy_lefts[i] + sxy_mid + sxy_rights[j]
            y_bar = sy / n

            m, b = linfit(sx, sx2, sy, sxy, n)

            # Sum of the squares of residuals
            ss_res = 0
            for k in range(lower_quarter - i - 1, upper_quarter + j + 1):
                ss_res += (y_data[k] - (m * x_data[k] + b)) ** 2

            # Sum of the squares of variance from mean
            ss_tot = sum((y - y_bar) ** 2 for y in y_data[lower_quarter - i - 1:upper_quarter + j + 1])

            # Definition of r-squared
            r2 = (1 - ss_res / ss_tot)

            # Save only useful values
            if r2 > r2_max:
                r2_max = r2
                i_max, j_max = i, j
                m_max, b_max = m, b

    return (lower_quarter - i_max - 1, upper_quarter + j_max + 1), (m_max, b_max), r2_max