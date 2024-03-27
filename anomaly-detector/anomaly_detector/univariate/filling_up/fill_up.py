# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from .fill_up_methods import *
from anomaly_detector.univariate.util.fields import DEFAULT_MAXIMUM_FILLUP_LENGTH


class FillingUpProcess:

    def __init__(self, indices, values, maximum_fillup_length=DEFAULT_MAXIMUM_FILLUP_LENGTH):
        if indices is None or not isinstance(indices, list):
            raise ValueError('indices is None or index is not a list')

        self.indices = indices
        self.init_values = values

        if len(indices) == 0:
            self.all_count = 0
            self.need_fill_up = False
            self.all_missing_count = 0
            self.missing_ratio = 0
            return
        if len(indices) != len(values):
            raise ValueError('length of indices is not aligned with length of values')

        self.need_fill_up = True
        self.all_missing_count = 0
        values_len = len(values)

        for i in range(values_len):
            if values[i] is None:
                raise ValueError('value at index %d is None' % indices[i])

        missing_found = False
        for i in range(values_len):
            if i != 0:
                if self.indices[i] <= self.indices[i - 1]:
                    raise ValueError('\'timestamp\' at index %d is out of order or duplicated or not aligned' % i)
                elif self.indices[i] > self.indices[i - 1] + 1:
                    missing_found = True
                    self.all_missing_count += self.indices[i] - self.indices[i - 1] - 1
        if self.indices[0] != 0:
            raise ValueError('indices[0] does not equal to 0')
        self.all_count = self.indices[-1] + 1  # assuming self.indices[0] is 0

        self.missing_ratio = self.all_missing_count / self.all_count
        if not missing_found or self.all_count > maximum_fillup_length:
            self.need_fill_up = False
        else:
            self.full_indices = list(range(self.all_count))
            self.missing_tags = [False] * self.all_count
            # how many values are missing continuously after this location, include the current one
            self.forward_missing_counts = [None] * self.all_count
            self.missing_counts = [None] * self.all_count  # count of this continuous missing segment
            j = 0
            cnt_missing = 0
            for i in self.full_indices:
                self.forward_missing_counts[i] = self.indices[j] - i
                if self.indices[j] == i:
                    j += 1
                    cnt_missing = 0
                elif self.indices[j] > i:
                    self.missing_tags[i] = True
                    if cnt_missing == 0:
                        cnt_missing = self.indices[j] - i
                self.missing_counts[i] = cnt_missing

    def __get_values_to_fill(self):
        values_to_fill = [None] * self.all_count
        for i in range(len(self.indices)):
            values_to_fill[self.indices[i]] = self.init_values[i]
        return values_to_fill

    def get_missing_indices(self):
        return [i for i in range(self.all_count) if self.forward_missing_counts[i] > 0]

    def get_missing_segments(self):
        missing_segment_indices = []
        missing_segment_lengths = []
        i = 0
        while i < self.all_count:
            if self.forward_missing_counts[i] > 0:
                missing_segment_indices.append(i)
                missing_segment_lengths.append(self.forward_missing_counts[i])
                i += self.forward_missing_counts[i]
            else:
                i += 1
        return missing_segment_indices, missing_segment_lengths

    def __step_by_step_fill_up(self, func, if_exception, **kwargs):
        if not self.need_fill_up:
            return None, None
        full_values = self.__get_values_to_fill()
        i = 0
        j = 0
        while i < self.all_count:
            if self.indices[j] == i:
                j += 1
                i += 1
            elif self.indices[j] > i:
                try:
                    res = func(partial_full_values=full_values, i=i, init_values=self.init_values, j=j,
                               forward_missing_counts=self.forward_missing_counts, missing_counts=self.missing_counts,
                               **kwargs)
                    if isinstance(res, list):
                        full_values[i:i + len(res)] = res
                        i += len(res)
                    else:
                        full_values[i] = res
                        i += 1
                except Exception as e:
                    if if_exception == 'raise':
                        raise e
                    elif if_exception == 'fill_with_none':
                        full_values[i] = None
                    elif if_exception == 'fill_with_last':
                        full_values[i] = func_last(init_values=self.init_values, j=j, **kwargs)
                    i += 1
        return full_values, self.missing_tags

    def __entire_fill_up(self, func, if_exception, **kwargs):
        try:
            full_values = func(self.init_values, self.indices, self.full_indices)
            return full_values, self.missing_tags
        except Exception as e:
            if if_exception == 'raise':
                raise e
            elif if_exception == 'fill_with_none':
                return self.__get_values_to_fill(), self.missing_tags
            elif if_exception == 'fill_with_last':
                return self.__step_by_step_fill_up(func_last, **kwargs)

    step_by_step_func_mapping = {
        'min': func_min,
        'max': func_max,
        'median': func_median,
        'average': func_average,
        'weighted_avg': func_weighted_avg,
        'constant': func_constant,
        'last': func_last,
        'linear_seg': func_linear_interpolation_seg,
        'spline_seg': func_spline_interpolation_seg
    }

    def fill_up(self, method: str, **kwargs):
        """
        :param method: str.
            {'auto', 'linear', 'spline', 'min', 'max', 'median', 'average', 'weighted_avg', 'constant', 'last',
            'linear_seg', 'spline_seg'}
        :param kwargs:
                backward_n: int.
                forward_n: int.
                number: number. used in constant method
                weights: list. used in weighted average method
                leverage_filled_values: bool, default False.
                consider_period: bool, default False. useless in interpolation methods.
                period: int, default 1. taking effects when consider_period is True or method is 'auto'.
                if_short_of_knowledge: str, default 'raise'. {'raise', 'fill_with_none', 'try_best', 'ignore'}
                if_exception: str, default 'try_best'. {'raise', 'fill_with_none', 'fill_with_last'}
        :return: full_values: list.
            list of value.
        :return: missing_tags: list.
            list of bool indicating whether the value is filled in this function or included at very beginning
        """

        if not self.need_fill_up:
            return None, None

        if 'if_exception' in kwargs:
            if kwargs['if_exception'] not in {'raise', 'fill_with_none', 'fill_with_last'}:
                raise Exception(
                    'invalid argument, if_exception should be one of "raise", "fill_with_none", "fill_with_last"')
        else:
            kwargs['if_exception'] = 'raise'

        if 'if_short_of_knowledge' in kwargs:
            if kwargs['if_short_of_knowledge'] not in {'raise', 'fill_with_none', 'try_best', 'ignore'}:
                raise Exception(
                    'invalid argument, if_short_of_knowledge should be one of "raise", "fill_with_none", "try_best", "ignore"')
        else:
            kwargs['if_short_of_knowledge'] = 'try_best'

        if 'period' in kwargs and kwargs['period'] is not None:
            if not isinstance(kwargs['period'], int) or kwargs['period'] < 0:
                raise ValueError('period is not int or period < 0')
            kwargs['period'] = max(kwargs['period'], 1)
        else:
            kwargs['period'] = 1

        if method == 'auto':
            return self.__auto_fill_up(**kwargs)

        elif method == 'linear':
            return self.__entire_fill_up(func_linear_interpolation_entire, **kwargs)

        elif method == 'spline':
            return self.__entire_fill_up(func_spline_interpolation_entire, **kwargs)

        elif method in FillingUpProcess.step_by_step_func_mapping.keys():
            return self.__step_by_step_fill_up(FillingUpProcess.step_by_step_func_mapping[method], **kwargs)

        else:
            raise NotImplementedError('filling up method "%s" is not supported yet' % method)

    def __scatter_distribution_fill_up(self, period):
        full_values = func_spline_interpolation_entire(self.init_values, self.indices, self.full_indices)
        max_value_global = np.max(self.init_values)
        min_value_global = np.min(self.init_values)
        i = 0
        j = 0
        while i < self.all_count:
            if self.indices[j] == i:
                j += 1
                i += 1
            elif self.indices[j] > i:
                if self.missing_counts[i] > 6 or (period > 1 and self.missing_counts[i] > period / 4):
                    full_values[i:i + self.missing_counts[i]] = [None] * self.missing_counts[i]
                    i += self.missing_counts[i]
                else:
                    # fix some desight
                    surrounding_4_real_values = \
                        [full_values[x] for x in
                         [i - 2, i - 1, i + self.missing_counts[i], i + self.missing_counts[i] + 1]
                         if 0 < x < self.all_count and not self.missing_tags[x]]
                    if len(surrounding_4_real_values) > 2 and \
                            (surrounding_4_real_values == sorted(surrounding_4_real_values)
                             or surrounding_4_real_values == sorted(surrounding_4_real_values, reverse=True)):
                        min_value = min(surrounding_4_real_values[0], surrounding_4_real_values[-1])
                        max_value = max(surrounding_4_real_values[0], surrounding_4_real_values[-1])
                        if any(full_values[x] < min_value or full_values[x] > max_value for x in
                               range(i, i + self.missing_counts[i])):
                            # linear interpolation
                            res = func_linear_interpolation_seg(missing_counts=self.missing_counts, i=i,
                                                                init_values=self.init_values, j=j)
                            full_values[i:i + len(res)] = res

                    # clip
                    for x in range(i, i + self.missing_counts[i]):
                        full_values[x] = min(max(full_values[x], min_value_global), max_value_global)

                    i += self.missing_counts[i]

        return full_values

    def __auto_fill_up(self, if_exception, period, **kwargs):
        try:
            if period > 1:  # seasonal

                # fill scatter
                filled = self.__scatter_distribution_fill_up(period=period)
                filled_indices = [self.full_indices[i] for i in range(self.all_count) if filled[i] is not None]
                filled = [x for x in filled if x is not None]

                # fill big gap
                process = FillingUpProcess(filled_indices, filled)
                if process.need_fill_up:
                    filled, tag = process.fill_up(method='weighted_avg', backward_n=3, forward_n=3,
                                                  consider_period=True, period=period,
                                                  weights=[0.1, 0.3, 0.6, 0.6, 0.3, 0.1],
                                                  if_exception='fill_with_none',
                                                  if_short_of_knowledge='ignore')
                    filled_indices = [self.full_indices[i] for i in range(self.all_count) if filled[i] is not None]
                    filled = [x for x in filled if x is not None]

                # fill remainder
                process = FillingUpProcess(filled_indices, filled)
                if process.need_fill_up:
                    filled, tag = process.fill_up(method='linear')

            else:  # non seasonal
                filled = func_linear_interpolation_entire(self.init_values, self.indices, self.full_indices)

            return filled, self.missing_tags

        except Exception as e:
            if if_exception == 'raise':
                raise e
            elif if_exception == 'fill_with_none':
                return self.__get_values_to_fill(), self.missing_tags
            elif if_exception == 'fill_with_last':
                return self.__step_by_step_fill_up(func_last, **kwargs)
