from scipy.stats import norm as Gaussian
from libc.math cimport sqrt, pow
from scipy.stats import t as student_t
import numpy as  np
cimport numpy as np
import array
cdef float Constant = Gaussian.ppf(3/4.)
import bisect
from libc.stdlib cimport rand

cdef float EPS = 1e-8

cpdef int partition(float[:] data, int left, int right, int pivot_index) nogil:
    cdef float pivot_value = data[pivot_index]
    cdef int left_index = left
    cdef int right_store_index = right
    cdef int i = left
    cdef int j
    while i <= right_store_index:
        if data[i] < pivot_value:
            data[left_index], data[i] = data[i], data[left_index]
            left_index += 1
        elif data[i] > pivot_value:
            data[right_store_index], data[i] = data[i], data[right_store_index]
            right_store_index -= 1
            i -= 1
        i += 1
    return (left_index + right_store_index) // 2

cpdef float select(float[:] data, int left, int right, int n) nogil:
    cdef int pivot_index
    if left == right:
        return data[left]
    cdef int rand_base = right - left + 1
    pivot_index = rand()% rand_base + left
    pivot_index = partition(data, left, right, pivot_index)
    # The pivot is in its final sorted position
    if n == pivot_index:
        return data[n]
    elif n < pivot_index:
        return select(data, left, pivot_index-1, n)
    else:
        return select(data, pivot_index+1, right, n)


cpdef float quick_select(float[:] data, int k) nogil:
    return select(data, 0, len(data)-1, k-1)

cpdef float fast_median(float[:] table) nogil:
    if len(table) % 2 == 1:
        return quick_select(table, len(table) // 2 + 1)
    else:
        return 0.5 * (quick_select(table, len(table) // 2) +
                      quick_select(table, len(table) // 2 + 1))

cpdef float calculate_esd_values(int i, float alpha, int n, bint one_tail=True):
    if one_tail:
        p = 1 - alpha / float(n - i + 1)
    else:
        p = 1 - alpha / float(2 * (n - i + 1))
    t = student_t.ppf(p, (n - i - 1))
    return t * (n - i) / float(sqrt((n - i - 1 + t ** 2) * (n - i + 1)))

cpdef float sorted_median(float[:] data, int i, int j):
    cdef int n = j - i
    cdef int mid
    if n == 0:
        raise Exception("no median for empty data")
    if n % 2 == 1:
        return data[i + n // 2]
    else:
        mid = i + n // 2
        return (data[mid - 1] + data[mid])/2

cpdef float find_median_sorted_arrays(float[:] a, float[:] b, float median):
    cdef int m = len(a)
    cdef int n = len(b)
    cdef int i_min = 0
    cdef int i_max = m
    cdef int half_len = (m + n + 1) // 2
    while i_min <= i_max:
        i = (i_min + i_max) // 2
        j = half_len - i
        if i < m and np.fabs(b[j - 1] - median) > np.fabs(a[i] - median):
            # i is too small, must increase it
            i_min = i + 1
        elif i > 0 and np.fabs(a[i - 1] - median) > np.fabs(b[j] - median):
            # i is too big, must decrease it
            i_max = i - 1
        else:
            # i is perfect
            if i == 0:
                max_of_left = np.fabs(b[j - 1] - median)
            elif j == 0:
                max_of_left = np.fabs(a[i - 1] - median)
            else:
                max_of_left = max(np.fabs(a[i - 1] - median), np.fabs(b[j - 1] - median))

            if (m + n) % 2 == 1:
                return max_of_left

            if i == m:
                min_of_right = np.fabs(b[j] - median)
            elif j == n:
                min_of_right = np.fabs(a[i] - median)
            else:
                min_of_right = min(np.fabs(a[i] - median), np.fabs(b[j] - median))

            return (max_of_left + min_of_right) / 2.0

cpdef bint check_anomaly_status(float median_value, float data_sigma, float value, float threshold,
                           bint upper_tail):
    cdef float de_median_value = value - median_value
    if not upper_tail:
        de_median_value = median_value - value
    if data_sigma < EPS:
        data_sigma = EPS
    de_median_value = de_median_value / data_sigma
    if de_median_value > threshold:
        return True
    return False


cpdef list dynamic_threshold(list sorted_values, list sorted_index, int max_outliers, float threshold,
                        bint upper_tail, int last_index):
    cdef int length = len(sorted_values)
    cdef int num_anoms = -1
    cdef int anomaly_index = -1
    cdef int start_index = 0
    cdef int k = 0
    cdef float median_value = 0
    cdef float data_sigma = 0
    cdef bint is_anomaly = False

    cdef float[:] values = array.array('f', sorted_values)
    cdef float[:] reverse_value = array.array('f', sorted_values[::-1])
    if last_index != -1:
        start_index = last_index

    for k in range(start_index, max_outliers):
        median_value = sorted_median(values, k, length)
        data_sigma = find_median_sorted_arrays(reverse_value[length - k - (length - k) // 2: length - k],
                                               values[k + (length - k) // 2:], median_value) / Constant

        is_anomaly = check_anomaly_status(median_value, data_sigma, values[k], threshold, upper_tail)

        if not is_anomaly:
            is_anomaly = check_anomaly_status(np.mean(values[k:length]),
                                              np.std(values[k:length]), values[k], threshold, upper_tail)

        if is_anomaly:
            num_anoms = k

        if is_anomaly and last_index != -1:
            break

    return sorted_index[:num_anoms + 1]


cpdef list generalized_esd_test(list sorted_values, list sorted_index, int max_outliers, list critical_values,
                           bint upper_tail, int last_index):
    cdef int length = len(sorted_values)
    cdef int num_anoms = -1
    cdef int anomaly_index = -1
    cdef int start_index = 0
    cdef int k = 0
    cdef float[:] values = array.array('f', sorted_values)
    cdef float[:] reverse_value = array.array('f', sorted_values[::-1])
    cdef float median_value = 0
    cdef float data_sigma = 0
    cdef bint is_anomaly = False
    if last_index != -1:
        start_index = last_index
    for k in range(start_index, max_outliers):
        median_value = sorted_median(values, k, length)
        data_sigma = find_median_sorted_arrays(reverse_value[length - k - (length - k) // 2: length - k],
                                               values[k + (length - k) // 2:], median_value) / Constant

        is_anomaly = check_anomaly_status(median_value, data_sigma, values[k], critical_values[k], upper_tail)

        if not is_anomaly:
            is_anomaly = check_anomaly_status(np.mean(values[k:length]),
                                              np.std(values[k:length]), values[k], critical_values[k], upper_tail)

        if is_anomaly:
            num_anoms = k

        if is_anomaly and last_index != -1:
            break

    return sorted_index[:num_anoms + 1]

cpdef max_gcv(np.ndarray data, np.ndarray periods):
    cdef float cv_mse = np.inf
    cdef float _mse = 0
    cdef float[:] _seasons = np.empty(1, dtype='f')
    cdef float[:] cv_seasons = np.empty(1, dtype='f')
    cdef int[:] periods_idx = periods
    cdef int i = 0
    for i in range(len(periods_idx)):
        _mse, _seasons = gcv(data, periods_idx[i])
        if _mse < cv_mse:
            cv_mse, cv_seasons = _mse, _seasons

    return cv_mse, cv_seasons

cpdef gcv(np.ndarray gcv_input, int period):
    cdef float[:] seasons = np.empty(period, dtype='f')
    cdef int[:] cycles = np.zeros(period, dtype='i')
    cdef float[:] sum_y2 = np.zeros(period, dtype='f')
    cdef float[:] sum_y = np.zeros(period, dtype='f')
    cdef int idx = 0
    cdef int i = 0
    cdef float[:] value_idx = array.array('f', [x for x in gcv_input])
    cdef float cv_mse = 0
    cdef int n = len(value_idx)
    for i in range(n):
        sum_y[idx] += value_idx[i]
        sum_y2[idx] += value_idx[i] * value_idx[i]
        cycles[idx] += 1
        idx = (i+1)%period

    for i in range(period):
        seasons[i] = sum_y[i] / cycles[i]
        cv_mse += (cycles[i] / (cycles[i] - 1.0)) ** 2 * (sum_y2[i] - sum_y[i] ** 2 / cycles[i])

    cv_mse = cv_mse / len(value_idx)
    cv_mse = 0.0 if np.isclose(cv_mse, 0.0) else cv_mse  # float precision noise
    return cv_mse, seasons

cpdef median_filter(np.ndarray data, int window, bint need_two_end=False):
    cdef int w_len = window // 2 * 2 + 1
    cdef int t_len = len(data)
    cdef float[:] val = array.array('f', [x for x in data])
    cdef float[:] ans = array.array('f', [x for x in data])
    cdef float[:] cur_windows = array.array('f', [0 for x in range(w_len)])
    cdef int delete_id
    cdef int add_id
    cdef int index
    if t_len < w_len:
        return ans
    for i in range(0, w_len):
        index = i
        add_id = bisect.bisect_right(cur_windows[:i], val[i])
        while index > add_id:
            cur_windows[index] = cur_windows[index - 1]
            index -= 1
        cur_windows[add_id] = data[i]
        if i >= w_len // 2 and need_two_end:
            ans[i - w_len // 2] = sorted_median(cur_windows, 0, i + 1)
    ans[window // 2] = sorted_median(cur_windows, 0, w_len)
    for i in range(window // 2 + 1, t_len - window // 2):
        delete_id = bisect.bisect_right(cur_windows, val[i - window // 2 - 1]) - 1
        index = delete_id
        while index < w_len - 1:
            cur_windows[index] = cur_windows[index + 1]
            index += 1

        add_id = bisect.bisect_right(cur_windows[:w_len - 1], val[i + window // 2])
        index = w_len - 1
        while index > add_id:
            cur_windows[index] = cur_windows[index - 1]
            index -= 1
        cur_windows[add_id] = data[i + window // 2]

        ans[i] = sorted_median(cur_windows, 0, w_len)

    if need_two_end:
        for i in range(t_len - window // 2, t_len):
            delete_id = bisect.bisect_right(cur_windows[: w_len], data[i - window // 2 - 1]) - 1
            index = delete_id
            while index < w_len - 1:
                cur_windows[index] = cur_windows[index + 1]
                index += 1
            w_len -= 1
            ans[i] = sorted_median(cur_windows[: w_len], 0, w_len)

    return ans


def spectral_residual_transform_core(list values):
    """
    This method transform a time series into spectral residual series
    :param values: list.
        a list of float values.
    :return: mag: list.
        a list of float values as the spectral residual values
    """

    cdef double[:] amplitude = array.array('d', [x for x in values])
    cdef double[:] mag = array.array('d', [x for x in values])
    cdef double current = 1.0
    cdef int window = 3


    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)

    ### the following lines are the same as the codes in comments
    # maglog = [np.log(item) if item != 0 else 0 for item in mag]
    # spectral = np.exp(maglog - average_filter(maglog, n=window))
    # trans.real = [ireal * ispectral / imag if imag != 0 else 0
    #               for ireal, ispectral, imag in zip(trans.real, spectral, mag)]
    # trans.imag = [iimag * ispectral / imag if imag != 0 else 0
    #               for iimag, ispectral, imag in zip(trans.imag, spectral, mag)]
    for i, d in enumerate(mag):
        if d > 0:
            current = current * d

        if i < window:
            amplitude[i] = pow(current, 1. / (i+1))
        else:
            if mag[i - window] > 0:
                current = current / mag[i - window]

            amplitude[i] = pow(current, 1. / window)

    trans.real = [ireal / a if a != 0 else 0 for ireal, a in zip(trans.real, amplitude)]
    trans.imag = [iimag / a if a != 0 else 0 for iimag, a in zip(trans.imag, amplitude)]
    # end of region

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)

    return mag


cpdef is_straight(np.ndarray values, int step):
    cdef int total_cnt = 0
    cdef int cnt = 0
    cdef double THRESHOLD = 1e-8
    for i in range(len(values)):
        if np.abs(np.var(values[i:min(len(values), i + step)])) - 0.0 < THRESHOLD:
            cnt += 1
        total_cnt += 1
    total_cnt = max(1, total_cnt // 2)
    return total_cnt <= cnt

cpdef remove_anomaly_in_bucket(np.ndarray values, int period):
    cdef float[:] sub_values_array
    for i in range(period):
        sub_values = values[i::period]
        sub_values_array = array.array('f', [x for x in sub_values])
        median = fast_median(sub_values_array)
        sub_values_array = array.array('f', [x for x in (sub_values - median)])
        mad = 1.4826 * fast_median(sub_values_array) + 1e-8
        spike_index = np.where((sub_values - median) / mad >= 3.0)[0]
        for k in range(len(spike_index)):
            values[spike_index[k] * period + i] = median
    return values
