# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import numpy as np
from scipy.interpolate import interp1d


def func_constant(number=None, **kwargs):
    return number


def func_last(init_values, j, **kwargs):
    return init_values[j - 1]


def _get_periodical_values(array, i, n, period, forward_missing_counts, missing_counts, leverage_filled_values,
                           if_short_of_knowledge, direction=-1):
    if period <= 0:
        raise Exception(f'invalid period {period}')
    if direction not in {1, -1}:
        raise Exception(f'invalid direction {direction}')

    result = []
    period *= direction
    i += period
    try_best = True if if_short_of_knowledge == 'try_best' else False
    while 0 <= i < len(array) and n > 0:
        if leverage_filled_values is False and forward_missing_counts[i] > 0:
            result.append(None)
            # accelerate
            if direction == 1:
                i += (int(np.ceil((forward_missing_counts[i] - period) / period))) * period
            else:
                i += (int(np.ceil((missing_counts[i] - forward_missing_counts[i]) / period))) * period * -1
        else:
            result.append(array[i])
            if array[i] is not None and try_best:
                n -= 1
        i += period
        if not try_best:
            n -= 1
    return result[::direction]


def __get_backward_and_forward_values(partial_full_values, i, init_values, j, period,
                                      forward_missing_counts, missing_counts, backward_n, forward_n,
                                      leverage_filled_values, if_short_of_knowledge):
    if leverage_filled_values is False and if_short_of_knowledge == 'try_best' and period == 1:
        # accelerate
        backward_values = init_values[j - min(backward_n, j): j]
        forward_values = init_values[j: j + forward_n]
    else:
        backward_values = _get_periodical_values(partial_full_values, i, backward_n, period, forward_missing_counts,
                                                 missing_counts, leverage_filled_values, if_short_of_knowledge,
                                                 direction=-1)
        forward_values = _get_periodical_values(partial_full_values, i, forward_n, period, forward_missing_counts,
                                                missing_counts, leverage_filled_values, if_short_of_knowledge,
                                                direction=1)

    if None in backward_values or len(backward_values) < backward_n or \
            None in forward_values or len(forward_values) < forward_n:
        if if_short_of_knowledge == 'raise':
            raise Exception('short of knowledge to fill up')
        elif if_short_of_knowledge == 'ignore' or if_short_of_knowledge == 'try_best':
            backward_values = [x for x in backward_values if x is not None]
            forward_values = [x for x in forward_values if x is not None]
            if len(backward_values) == 0 and len(forward_values) == 0:
                return None, None
        elif if_short_of_knowledge == 'fill_with_none':
            return None, None
    return backward_values, forward_values


def __basic_func(partial_full_values, i, init_values, j, forward_missing_counts, missing_counts, func,
                 backward_n, forward_n=0, leverage_filled_values=False, consider_period=False, period=None,
                 if_short_of_knowledge='try_best', weights=None, **kwargs):
    if not (consider_period and period is not None and period > 1):
        period = 1
    backward_values, forward_values = __get_backward_and_forward_values(partial_full_values, i, init_values, j, period,
                                                                        forward_missing_counts, missing_counts,
                                                                        backward_n, forward_n,
                                                                        leverage_filled_values, if_short_of_knowledge)
    if backward_values is None:
        return None

    if func is np.average and weights is not None:
        if len(weights) != (backward_n + forward_n):
            raise Exception('size of weights is not aligned with (backward_n + forward_n)')
        covered_weights = weights[backward_n - len(backward_values):backward_n + len(forward_values)]
        return func(backward_values + forward_values, weights=covered_weights)
    else:
        return func(backward_values + forward_values)


def func_average(**kwargs):
    return __basic_func(func=np.average, **kwargs)


def func_weighted_avg(weights, **kwargs):
    return __basic_func(func=np.average, weights=weights, **kwargs)


def func_median(**kwargs):
    return __basic_func(func=np.median, **kwargs)


def func_min(**kwargs):
    return __basic_func(func=np.min, **kwargs)


def func_max(**kwargs):
    return __basic_func(func=np.max, **kwargs)


def func_linear_interpolation(i, init_values, j, forward_missing_counts, missing_counts, **kwargs):
    values = func_linear_interpolation_seg(missing_counts=missing_counts, i=i, init_values=init_values, j=j, **kwargs)
    return values[missing_counts[i] - forward_missing_counts[i]]


def func_linear_interpolation_seg(missing_counts, i, init_values, j, **kwargs):
    f = interp1d([0, 1], init_values[j - 1:j + 1])
    values = f(np.linspace(0, 1, num=missing_counts[i] + 2, endpoint=True))[1:-1].tolist()
    return values


def func_linear_interpolation_entire(init_values, indices, full_indices, **kwargs):
    f = interp1d(indices, init_values)
    values = f(full_indices).tolist()
    return values


def func_spline_interpolation(i, forward_missing_counts, missing_counts, **kwargs):
    values = func_spline_interpolation_seg(i=i, forward_missing_counts=forward_missing_counts,
                                           missing_counts=missing_counts, **kwargs)
    return values[missing_counts[i] - forward_missing_counts[i]]


def func_spline_interpolation_seg(partial_full_values, i, init_values, j, forward_missing_counts, missing_counts,
                                  backward_n, forward_n, leverage_filled_values=False,
                                  if_short_of_knowledge='try_best', **kwargs):
    backward_values, forward_values = __get_backward_and_forward_values(partial_full_values, i, init_values, j, 1,
                                                                        forward_missing_counts, missing_counts,
                                                                        backward_n, forward_n,
                                                                        leverage_filled_values, if_short_of_knowledge)

    input_values = backward_values + forward_values
    pre_count = len(backward_values)
    post_count = len(forward_values)

    input_indices = list(range(pre_count)) + list(
        range(missing_counts[i] + pre_count, missing_counts[i] + pre_count + post_count))
    try:
        f = interp1d(input_indices, input_values, kind='cubic')
        values = f(range(missing_counts[i] + pre_count + post_count))[pre_count:-post_count].tolist()
        return values
    except ValueError:
        raise Exception('not enough point to do spline interpolation, at least 4 points')


def func_spline_interpolation_entire(init_values, indices, full_indices, **kwargs):
    if len(init_values) <= 3:
        raise Exception('not enough point to do spline interpolation, at least 4 points')
    f = interp1d(indices, init_values, kind='cubic')
    values = f(full_indices).tolist()
    return values
