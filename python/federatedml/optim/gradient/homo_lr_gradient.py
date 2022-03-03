#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import functools

import numpy as np
import scipy as sp

from federatedml.feature.sparse_vector import SparseVector
from federatedml.optim.gradient.hetero_linear_model_gradient import HeteroGradientBase
from federatedml.statistic import data_overview
from federatedml.util import LOGGER
from federatedml.util import fate_operator
from federatedml.util.fate_operator import vec_dot
from federatedml.util.fixpoint_solver import FixedPointEncoder


def load_data(data_instance):
    X = []
    Y = []
    for iter_key, instant in data_instance:
        if instant.weight is not None:
            weighted_feature = instant.weight * instant.features
        else:
            weighted_feature = instant.features
        X.append(weighted_feature)
        if instant.label == 1:
            Y.append([1])
        else:
            Y.append([-1])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


class LogisticGradient(object):

    @staticmethod
    def compute_loss(values, coef, intercept):
        X, Y = load_data(values)
        batch_size = len(X)
        if batch_size == 0:
            LOGGER.warning("This partition got 0 data")
            return None
        tot_loss = np.log(1 + np.exp(np.multiply(-Y.transpose(), X.dot(coef) + intercept))).sum()
        return tot_loss

    @staticmethod
    def compute_gradient(values, coef, intercept, fit_intercept):
        X, Y = load_data(values)
        batch_size = len(X)
        if batch_size == 0:
            LOGGER.warning("This partition got 0 data")
            return None

        d = (1.0 / (1 + np.exp(-np.multiply(Y.transpose(), X.dot(coef) + intercept))) - 1).transpose() * Y
        grad_batch = d * X
        if fit_intercept:
            grad_batch = np.c_[grad_batch, d]
        grad = sum(grad_batch)
        return grad

    @staticmethod
    def compute_d(data_instances, w):
        d = data_instances.mapValues(
            lambda v: vec_dot(v.features, w.coef_) + w.intercept_ - v.label)
        return d

    
    def compute_linr_gredient(self,data_instances,fore_gradient,fit_intercept):
        is_sparse = data_overview.is_sparse_data(data_instances)
        data_count = data_instances.count()
        fixed_point_encoder = FixedPointEncoder(2 ** 23)
        feat_join_grad = data_instances.join(fore_gradient,
                                             lambda d, g: (d.features, g))
        f = functools.partial(self.__apply_cal_gradient,
                              fixed_point_encoder=fixed_point_encoder,
                              is_sparse=is_sparse)
        gradient_sum = feat_join_grad.applyPartitions(f)
        gradient_sum = gradient_sum.reduce(lambda x, y: x + y)
        if fit_intercept:
            # bias_grad = np.sum(fore_gradient)
            bias_grad = fore_gradient.reduce(lambda x, y: x + y)
            gradient_sum = np.append(gradient_sum, bias_grad)
        gradient = gradient_sum / data_count
        return gradient

    @staticmethod
    def __apply_cal_gradient(data, fixed_point_encoder, is_sparse):
        all_g = None
        for key, (feature, d) in data:
            if is_sparse:
                x = np.zeros(feature.get_shape())
                for idx, v in feature.get_all_data():
                    x[idx] = v
                feature = x
            if fixed_point_encoder:
                # g = (feature * 2 ** floating_point_precision).astype("int") * d
                g = fixed_point_encoder.encode(feature) * d
            else:
                g = feature * d
            if all_g is None:
                all_g = g
            else:
                all_g += g
        if all_g is None:
            return all_g
        elif fixed_point_encoder:
            all_g = fixed_point_encoder.decode(all_g)
        return all_g

class TaylorLogisticGradient(object):

    @staticmethod
    def compute_gradient(values, coef, intercept, fit_intercept):
        LOGGER.debug("Get in compute_gradient")
        X, Y = load_data(values)
        batch_size = len(X)
        if batch_size == 0:
            return None

        one_d_y = Y.reshape([-1, ])
        d = (0.25 * np.array(fate_operator.dot(X, coef) + intercept).transpose() + 0.5 * one_d_y * -1)

        grad_batch = X.transpose() * d
        grad_batch = grad_batch.transpose()
        if fit_intercept:
            grad_batch = np.c_[grad_batch, d]
        grad = sum(grad_batch)
        LOGGER.debug("Finish compute_gradient")
        return grad
