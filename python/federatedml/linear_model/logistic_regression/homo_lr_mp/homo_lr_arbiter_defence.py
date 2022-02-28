#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from copy import deepcopy
import itertools
from typing import Iterable, overload
import numpy as np
from functools import reduce
import functools

from numpy.core import overrides
from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.homo.procedure import paillier_cipher
from federatedml.linear_model.linear_model_weight import LinearModelWeights as LogisticRegressionWeights
from federatedml.linear_model.logistic_regression.homo_logistic_regression.homo_lr_base import HomoLRBase
from federatedml.optim import activation
from federatedml.framework.weights import Weights, NumericWeights, TransferableWeights
from federatedml.util import fate_operator
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.param.lr_poison_param import PoisonParam

import torch
from federatedml.optim.gradient.homo_lr_gradient import LogisticGradient
from federatedml.model_selection import MiniBatch


class ArbiterWithDefence(aggregator.Arbiter):

    def __init__(self, trans_var=None):
        super().__init__(trans_var=trans_var)
        self.clients_models = None
        self.detect = False

    def get_party_info(self, index: int):
        return self._client_parties[index]

    def init_detect(self, detect, detect_tol, detect_rule, optimizer, detect_data):
        self.detect = detect
        self.detect_tol = detect_tol
        self.detect_rule = detect_rule
        self.optimizer = optimizer
        self.detect_data = detect_data

    def check_models_by_weights(self, models, suffix):
        weight_list = []
        for weight, degree in models:
            weight = deepcopy(weight)
            weight /= degree
            weight_list.append(weight.unboxed)
        client_num = len(weight_list)
        abnormal_list = []
        for i in range(client_num):
            client_i = self._client_parties[i]
            wi = torch.from_numpy(weight_list[i])
            for j in range(i + 1, client_num):
                client_j = self._client_parties[j]
                wj = torch.from_numpy(weight_list[j])
                sim = torch.cosine_similarity(wi, wj, dim=0)
                if sim.item() < self.detect_tol:
                    LOGGER.warning(
                        '\nat iter {}, weight similarity between {} and {} is {}, lower than tolence{}'
                            .format(suffix, client_i, client_j, sim.item(), self.detect_tol))
                    abnormal_list.append({client_i, client_j})
                print("cosine similarity between {} and {} is {}".format(
                    client_i, client_j, sim.item()))
        if len(abnormal_list) != 0:
            abnormal_set = reduce(lambda x, y: x & y, abnormal_list)
            LOGGER.warning('\nmay be poisoned by {} at iter{}'.format(abnormal_set, suffix))
        pass

    def check_model_by_loss(self, client_models, detect_data, suffix):
        weight_list = []
        degree_list = []
        agg_drgree = 0
        for weight, degree in client_models:
            weight_list.append(weight)
            degree_list.append(degree)
            agg_drgree += degree
        agg_model = reduce(lambda x, y: x + y, weight_list)
        # agg_drgree = reduce(lambda x, y: x + y, degree_list)
        agg_model /= agg_drgree
        f_loss = functools.partial(LogisticGradient.compute_loss,
                                   coef=agg_model.coef_,
                                   intercept=agg_model.intercept_)
        agg_loss = detect_data.applyPartitions(f_loss).reduce(fate_operator.reduce_add)
        loss_norm = self.optimizer.loss_norm(agg_model)
        if loss_norm is not None:
            agg_loss += loss_norm
        agg_loss /= detect_data.count()
        model_list = list(map(lambda x, y: x / y, weight_list, degree_list))
        for index, model in enumerate(model_list):
            client = self._client_parties[index]
            f_loss = functools.partial(LogisticGradient.compute_loss,
                                       coef=model.coef_,
                                       intercept=model.intercept_)
            loss = detect_data.applyPartitions(f_loss).reduce(fate_operator.reduce_add)
            loss_norm = self.optimizer.loss_norm(model)
            if loss_norm is not None:
                loss += loss_norm
            loss /= detect_data.count()
            if loss - agg_loss > self.detect_tol:
                LOGGER.warning(
                    "\nAt iter {},may be poisoned by {}, his model loss is {}, the aggregated model loss is {}, the dirrerence is greater than{}"
                        .format(suffix, client, loss, agg_loss, self.detect_tol))
            print('At iter{}, the loss of model from {} is {}'.format(suffix, client, loss))
        pass

    def init_defence(self, defence, defence_tol, defence_rule):
        self.defence = defence
        self.defence_tol = defence_tol
        self.defence_rule = defence_rule

    def get_clients_models(self):
        return self.clients_models

    def loss_function_rejection(self, total_model, total_degree, defence_models):
        # model_a is the aggregated model with all parties
        model_a = total_model / total_degree
        f = functools.partial(LogisticGradient.compute_loss,
                              coef=(model_a).coef_,
                              intercept=(model_a).intercept_)
        loss_a = self.validate_data.applyPartitions(f).reduce(fate_operator.reduce_add)
        loss_norm = self.optimizer.loss_norm(model_a)
        if loss_norm is not None:
            loss_a += loss_norm
        loss_a /= self.validate_data.count()
        print('total model loss {}'.format(loss_a))
        # loss_impact = loss_a - loss_b
        # indicate the loss caused by the model_b(from one party)
        loss_impact = []
        model_list = []
        degree_list = []
        for model, degree in defence_models:
            model_list.append(deepcopy(model))
            degree_list.append(degree)
            # model_b is the model that removed one party
            model_b = (total_model - model)
            degree_b = (total_degree - degree)
            model_b /= degree_b
            f = functools.partial(LogisticGradient.compute_loss,
                                  coef=(model_b).coef_,
                                  intercept=(model_b).intercept_)
            loss_b = self.validate_data.applyPartitions(f).reduce(fate_operator.reduce_add)
            loss_norm = self.optimizer.loss_norm(model_b)
            if loss_b is not None:
                loss_b += loss_norm
            loss_b /= self.validate_data.count()
            loss_impact.append(loss_a - loss_b)

        import heapq
        # get the i-th largest loss impact
        remove_indexs = map(loss_impact.index, heapq.nlargest(self.defence_tol, loss_impact))
        remove_indexs = list(remove_indexs)
        LOGGER.warning('\n remore indexes{}'.format(remove_indexs))
        for index in remove_indexs:
            total_model -= model_list[index]
            total_degree -= degree_list[index]
        total_model /= total_degree
        return total_model

    # override
    def aggregate_model(self, ciphers_dict=None, suffix=tuple()) -> Weights:
        models = self.get_models_for_aggregate(ciphers_dict, suffix=suffix)
        models, detect_models, defence_models = itertools.tee(models, 3)
        if self.detect:
            if self.detect_rule == 'weight_similarity':
                self.check_models_by_weights(detect_models, suffix)
            elif self.detect_rule == 'loss':
                self.check_model_by_loss(client_models=detect_models,
                                         detect_data=self.detect_data,
                                         suffix=suffix)
        total_model, total_degree = reduce(lambda x, y: (x[0] + y[0], x[1] + y[1]), models)
        if self.defence:
            if self.defence_rule == "lfr":
                return self.loss_function_rejection(total_model, total_degree, defence_models)
        total_model /= total_degree
        LOGGER.debug("In aggregate model, total_model: {}, total_degree: {}".format(
            total_model.unboxed, total_degree))
        return total_model


class HomoLRArbiter(HomoLRBase):

    def __init__(self):
        super(HomoLRArbiter, self).__init__()
        self.re_encrypt_times = []  # Record the times needed for each host

        self.loss_history = []
        self.is_converged = False
        self.role = consts.ARBITER
        # self.aggregator = aggregator.Arbiter()
        self.aggregator = ArbiterWithDefence()
        self.model_weights = None
        self.cipher = paillier_cipher.Arbiter()
        self.host_predict_results = []
        self.gradient_operator = LogisticGradient()
        self.model_param = PoisonParam()

    def _init_model(self, params):
        super()._init_model(params)
        self.cipher.register_paillier_cipher(self.transfer_variable)
        # detect
        self.detect = params.detect
        self.detect_tol = params.detect_tol
        self.detect_rule = params.detect_rule
        # defence
        self.defence = params.defence
        self.defence_rule = params.defence_rule
        self.aggregator.init_defence(self.defence, params.defence_tol, params.defence_rule)

    def check_model_by_loss(self, models, detect_data, agg_model_loss):
        weight_list = []
        for weight, degree in models:
            weight = deepcopy(weight)
            weight /= degree
            weight_list.append(weight)
        for index, model in enumerate(weight_list):
            client = self.aggregator._client_parties[index]
            f_loss = functools.partial(LogisticGradient.compute_loss,
                                       coef=model.coef_,
                                       intercept=model.intercept_)
            loss = detect_data.applyPartitions(f_loss).reduce(fate_operator.reduce_add)
            loss_norm = self.optimizer.loss_norm(model)
            if loss_norm is not None:
                loss += loss_norm
            loss /= detect_data.count()
            if loss - agg_model_loss > self.detect_tol:
                LOGGER.warning(
                    "\nAt iter {},may be poisoned by {}, his model loss is {}, the aggregated model loss is {}, the dirrerence is greater than{}"
                        .format(self.n_iter_, client, loss, agg_model_loss, self.detect_tol))
            print('At iter{}, the loss of model from {} is {}'.format(self.n_iter_, client, loss))
        pass

    def check_models_by_weights(self, models):
        weight_list = []
        for weight, degree in models:
            weight = deepcopy(weight)
            weight /= degree
            weight_list.append(weight.unboxed)
        client_num = len(weight_list)
        abnormal_list = []
        for i in range(client_num):
            client_i = self.aggregator._client_parties[i]
            wi = torch.from_numpy(weight_list[i])
            for j in range(i + 1, client_num):
                client_j = self.aggregator._client_parties[j]
                wj = torch.from_numpy(weight_list[j])
                sim = torch.cosine_similarity(wi, wj, dim=0)
                if sim.item() < self.detect_tol:
                    LOGGER.warning(
                        '\nat iter {}, weight similarity between {} and {} is {}, lower than tolence{}'
                            .format(self.n_iter_, client_i, client_j, sim.item(), self.detect_tol))
                    abnormal_list.append({client_i, client_j})
                print("cosine similarity between {} and {} is {}".format(
                    client_i, client_j, sim.item()))
        if len(abnormal_list) != 0:
            abnormal_set = reduce(lambda x, y: x & y, abnormal_list)
            LOGGER.warning('\nmay be poisoned by {} at iter{}'.format(abnormal_set, self.n_iter_))
        pass

    def fit(self, data_instances=None, validate_data=None):
        self._server_check_data()

        host_ciphers = self.cipher.paillier_keygen(
            key_length=self.model_param.encrypt_param.key_length, suffix=('fit',))
        host_has_no_cipher_ids = [idx for idx, cipher in host_ciphers.items() if cipher is None]
        self.re_encrypt_times = self.cipher.set_re_cipher_time(host_ciphers)
        max_iter = self.max_iter
        # validation_strategy = self.init_validation_strategy()
        if data_instances:
            self._abnormal_detection(data_instances)
            self.check_abnormal_values(data_instances)
            self.init_schema(data_instances)
            self.model_weights = self._init_model_variables(data_instances)
        if self.detect:
            self.aggregator.init_detect(self.detect, self.detect_tol, self.detect_rule,
                                        self.optimizer, data_instances)
        while self.n_iter_ < max_iter + 1:
            suffix = (self.n_iter_,)

            if ((self.n_iter_ + 1) % self.aggregate_iters == 0) or self.n_iter_ == max_iter:

                merged_model = self.aggregator.aggregate_and_broadcast(ciphers_dict=host_ciphers,
                                                                       suffix=suffix)

                total_loss = self.aggregator.aggregate_loss(host_has_no_cipher_ids, suffix)
                self.callback_loss(self.n_iter_, total_loss)
                self.loss_history.append(total_loss)

                if self.use_loss:
                    converge_var = total_loss
                else:
                    converge_var = np.array(merged_model.unboxed)

                self.is_converged = self.aggregator.send_converge_status(
                    self.converge_func.is_converge, (converge_var,), suffix=(self.n_iter_,))
                LOGGER.info("n_iters: {}, total_loss: {}, converge flag is :{}".format(
                    self.n_iter_, total_loss, self.is_converged))
                self.prev_round_weights = deepcopy(self.model_weights)
                self.model_weights = LogisticRegressionWeights(
                    merged_model.unboxed, self.model_param.init_param.fit_intercept)
                if data_instances:
                    loss = self._compute_loss(data_instances, self.prev_round_weights)
                    print("loss{}".format(loss))
                # if self.detect:
                #     if self.detect_rule == 'loss':
                #         self.check_model_by_loss(self.aggregator.get_clients_models(),
                #                                  data_instances, loss)
                #     elif self.detect_rule == "weight_similarity":
                #         self.check_models_by_weights(self.aggregator.get_clients_models())
                if self.header is None:
                    self.header = ['x' + str(i) for i in range(len(self.model_weights.coef_))]

                if self.is_converged or self.n_iter_ == max_iter:
                    break

            self.cipher.re_cipher(iter_num=self.n_iter_,
                                  re_encrypt_times=self.re_encrypt_times,
                                  host_ciphers_dict=host_ciphers,
                                  re_encrypt_batches=self.re_encrypt_batches)

            # validation_strategy.validate(self, self.n_iter_)
            self.n_iter_ += 1

        LOGGER.info("Finish Training task, total iters: {}".format(self.n_iter_))

    def predict(self, data_instantces=None):
        LOGGER.info(f'Start predict task')
        current_suffix = ('predict',)
        host_ciphers = self.cipher.paillier_keygen(
            key_length=self.model_param.encrypt_param.key_length, suffix=current_suffix)

        # LOGGER.debug("Loaded arbiter model: {}".format(self.model_weights.unboxed))
        for idx, cipher in host_ciphers.items():
            if cipher is None:
                continue
            encrypted_model_weights = self.model_weights.encrypted(cipher, inplace=False)
            self.transfer_variable.aggregated_model.remote(obj=encrypted_model_weights.for_remote(),
                                                           role=consts.HOST,
                                                           idx=idx,
                                                           suffix=current_suffix)

        # Receive wx results

        for idx, cipher in host_ciphers.items():
            if cipher is None:
                continue
            encrypted_predict_wx = self.transfer_variable.predict_wx.get(idx=idx,
                                                                         suffix=current_suffix)
            predict_wx = cipher.distribute_decrypt(encrypted_predict_wx)

            prob_table = predict_wx.mapValues(lambda x: activation.sigmoid(x))
            predict_table = prob_table.mapValues(
                lambda x: 1 if x > self.model_param.predict_param.threshold else 0)

            self.transfer_variable.predict_result.remote(predict_table,
                                                         role=consts.HOST,
                                                         idx=idx,
                                                         suffix=current_suffix)
            self.host_predict_results.append((prob_table, predict_table))
