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

import functools

import numpy as np
import scipy.sparse as sp

from federatedml.feature.sparse_vector import SparseVector
from federatedml.statistic import data_overview
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.util.fixpoint_solver import FixedPointEncoder


class HeteroGradientBase(object):
    def __init__(self):
        self.use_async = False
        self.use_sample_weight = False
        self.fixed_point_encoder = None

    def compute_gradient_procedure(self, *args):
        raise NotImplementedError("Should not call here")

    def set_total_batch_nums(self, total_batch_nums):
        """
        Use for sqn gradient.
        """
        pass

    def set_use_async(self):
        self.use_async = True

    def set_use_sync(self):
        self.use_async = False

    def set_use_sample_weight(self):
        self.use_sample_weight = True

    def set_fixed_float_precision(self, floating_point_precision):
        if floating_point_precision is not None:
            self.fixed_point_encoder = FixedPointEncoder(2**floating_point_precision)

    # 执行的 data 值元组中两值乘法并累加求和
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

    def compute_gradient(self, data_instances, fore_gradient, fit_intercept, need_average=True):
        """
        Compute hetero-regression gradient
        Parameters
        ----------
        data_instances: Table, input data
        fore_gradient: Table, fore_gradient
        fit_intercept: bool, if model has intercept or not
        need_average: bool, gradient needs to be averaged or not

        Returns
        ----------
        Table
            the hetero regression model's gradient
        """

        # feature_num = data_overview.get_features_shape(data_instances)
        # data_count = data_instances.count()
        is_sparse = data_overview.is_sparse_data(data_instances)

        LOGGER.debug("Use apply partitions")
        feat_join_grad = data_instances.join(fore_gradient,
                                             lambda d, g: (d.features, g))
        # 利用偏函数定义部分参数
        f = functools.partial(self.__apply_cal_gradient,
                              fixed_point_encoder=self.fixed_point_encoder,
                              is_sparse=is_sparse)

        # 根据前向传播的 ∂l/∂w = ∂l/∂y*∂y/∂w ，其中 ∂l/∂y 就是 fore_gradient，而 ∂y/∂w 等于 x，就是对应的输入值
        gradient_sum = feat_join_grad.applyPartitions(f)

        gradient_sum = gradient_sum.reduce(lambda x, y: x + y)
        if fit_intercept:
            # bias_grad = np.sum(fore_gradient)
            bias_grad = fore_gradient.reduce(lambda x, y: x + y)
            gradient_sum = np.append(gradient_sum, bias_grad)

        if need_average:
            gradient = gradient_sum / data_instances.count()
        else:
            gradient = gradient_sum

        """
        else:
            LOGGER.debug(f"Original_method")
            feat_join_grad = data_instances.join(fore_gradient,
                                                 lambda d, g: (d.features, g))
            f = functools.partial(self.__compute_partition_gradient,
                                  fit_intercept=fit_intercept,
                                  is_sparse=is_sparse)
            gradient_partition = feat_join_grad.applyPartitions(f)
            gradient_partition = gradient_partition.reduce(lambda x, y: x + y)

            gradient = gradient_partition / data_count
        """
        return gradient


class Guest(HeteroGradientBase):
    def __init__(self):
        super().__init__()
        self.half_d = None
        self.host_forwards = None
        self.forwards = None
        self.aggregated_forwards = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                guest_gradient_transfer, guest_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = guest_gradient_transfer
        self.unilateral_optim_gradient_transfer = guest_optim_gradient_transfer

    def compute_and_aggregate_forwards(self, data_instances, model_weights,
                                       cipher, batch_index, current_suffix, offset=None):
        raise NotImplementedError("Function should not be called here")

    def compute_half_d(self, data_instances, w, cipher, batch_index, current_suffix):
        raise NotImplementedError("Function should not be called here")

    def _asynchronous_compute_gradient(self, data_instances, model_weights, cipher, current_suffix):
        LOGGER.debug("Called asynchronous gradient")
        encrypted_half_d = cipher.distribute_encrypt(self.half_d)
        self.remote_fore_gradient(encrypted_half_d, suffix=current_suffix)

        half_g = self.compute_gradient(data_instances, self.half_d, False)
        self.host_forwards = self.get_host_forward(suffix=current_suffix)
        host_forward = self.host_forwards[0]
        host_half_g = self.compute_gradient(data_instances, host_forward, False)
        unilateral_gradient = half_g + host_half_g
        if model_weights.fit_intercept:
            n = data_instances.count()
            intercept = (host_forward.reduce(lambda x, y: x + y) + self.half_d.reduce(lambda x, y: x + y)) / n
            unilateral_gradient = np.append(unilateral_gradient, intercept)
        return unilateral_gradient

    def _centralized_compute_gradient(self, data_instances, model_weights, cipher, current_suffix, masked_index=None):
        # 获取所有 Host 前向传播的结果
        self.host_forwards = self.get_host_forward(suffix=current_suffix)

        # 利用 compute_half_d 计算出的 Guest 前向传播与目标之间的差值
        fore_gradient = self.half_d

        batch_size = data_instances.count()
        partial_masked_index_enc = None
        if masked_index:
            masked_index = masked_index.mapValues(lambda value: 0)
            masked_index_to_encrypt = masked_index.subtractByKey(self.half_d)
            partial_masked_index_enc = cipher.distribute_encrypt(masked_index_to_encrypt)

        # 累加后获得所有 Host + Guest 前向传播之和与目标之间的差值，即 d = Σ(W_Hj * X_Hj) + W_G * X_G - Y
        for host_forward in self.host_forwards:
            if self.use_sample_weight:
                # host_forward = host_forward.join(data_instances, lambda h, v: h * v.weight)
                host_forward = data_instances.join(host_forward, lambda v, h: h * v.weight)
            fore_gradient = fore_gradient.join(host_forward, lambda x, y: x + y)

        def _apply_obfuscate(val):
            val.apply_obfuscator()
            return val
        fore_gradient = fore_gradient.mapValues(lambda val: _apply_obfuscate(val) / batch_size)

        # 将所有参与方前向传播之和与目标之间的差值传递给 Host 参与方
        if partial_masked_index_enc:
            masked_fore_gradient = partial_masked_index_enc.union(fore_gradient)
            self.remote_fore_gradient(masked_fore_gradient, suffix=current_suffix)
        else:
            self.remote_fore_gradient(fore_gradient, suffix=current_suffix)

        # self.remote_fore_gradient(fore_gradient, suffix=current_suffix)
        # 计算获得 w 权重参数对应的梯度
        unilateral_gradient = self.compute_gradient(data_instances, fore_gradient,
                                                    model_weights.fit_intercept, need_average=False)
        return unilateral_gradient

    def compute_gradient_procedure(self, data_instances, cipher, model_weights, optimizer,
                                   n_iter_, batch_index, offset=None, masked_index=None):
        """
          Linear model gradient procedure
          Step 1: get host forwards which differ from different algorithm
                  For Logistic Regression and Linear Regression: forwards = wx
                  For Poisson Regression, forwards = exp(wx)

          Step 2: Compute self forwards and aggregate host forwards and get d = fore_gradient

          Step 3: Compute unilateral gradient = ∑d*x,

          Step 4: Send unilateral gradients to arbiter and received the optimized and decrypted gradient.
          """
        current_suffix = (n_iter_, batch_index)
        # self.host_forwards = self.get_host_forward(suffix=current_suffix)

        # Compute Guest's partial d
        # 计算 Guest 本地模型前向传播结果与目标之间的差值，即 W_G * X_G - y， 保存至 self.half_d
        self.compute_half_d(data_instances, model_weights, cipher,
                            batch_index, current_suffix)
        if self.use_async:
            unilateral_gradient = self._asynchronous_compute_gradient(data_instances, model_weights,
                                                                      cipher=cipher,
                                                                      current_suffix=current_suffix)
        else:
            unilateral_gradient = self._centralized_compute_gradient(data_instances, model_weights,
                                                                     cipher=cipher,
                                                                     current_suffix=current_suffix,
                                                                     masked_index=masked_index)

        if optimizer is not None:
            unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)

        # 将加密的梯度发送给 Arbiter，然后获取解密后的梯度
        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        # LOGGER.debug(f"Before return, optimized_gradient: {optimized_gradient}")
        return optimized_gradient

    # 获取 Host 前向传播的结果
    def get_host_forward(self, suffix=tuple()):
        host_forward = self.host_forward_transfer.get(idx=-1, suffix=suffix)
        return host_forward

    # 将前向传播与目标之间的差值传递给 Host
    def remote_fore_gradient(self, fore_gradient, suffix=tuple()):
        self.fore_gradient_transfer.remote(obj=fore_gradient, role=consts.HOST, idx=-1, suffix=suffix)

    # 将加密的梯度发送给 Arbiter，然后获取解密后的梯度
    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Host(HeteroGradientBase):
    def __init__(self):
        super().__init__()
        self.forwards = None
        self.fore_gradient = None

    def _register_gradient_sync(self, host_forward_transfer, fore_gradient_transfer,
                                host_gradient_transfer, host_optim_gradient_transfer):
        self.host_forward_transfer = host_forward_transfer
        self.fore_gradient_transfer = fore_gradient_transfer
        self.unilateral_gradient_transfer = host_gradient_transfer
        self.unilateral_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_forwards(self, data_instances, model_weights):
        raise NotImplementedError("Function should not be called here")

    def compute_unilateral_gradient(self, data_instances, fore_gradient, model_weights, optimizer):
        raise NotImplementedError("Function should not be called here")

    def _asynchronous_compute_gradient(self, data_instances, cipher, current_suffix):
        encrypted_forward = cipher.distribute_encrypt(self.forwards)
        self.remote_host_forward(encrypted_forward, suffix=current_suffix)

        half_g = self.compute_gradient(data_instances, self.forwards, False)
        guest_half_d = self.get_fore_gradient(suffix=current_suffix)
        guest_half_g = self.compute_gradient(data_instances, guest_half_d, False)
        unilateral_gradient = half_g + guest_half_g
        return unilateral_gradient

    def _centralized_compute_gradient(self, data_instances, cipher, current_suffix):
        # 对 Host 前向传播的结果执行同态加密
        encrypted_forward = cipher.distribute_encrypt(self.forwards)

        # 将加密后 Host 前向传播结果发送给 Guest
        self.remote_host_forward(encrypted_forward, suffix=current_suffix)

        # 获取所有参与方前向传播的结果与目标之间的差值 d = Σ(W_Hj * X_Hj) + W_G * X_G - Y，可以理解为全局 y 对应的梯度
        fore_gradient = self.fore_gradient_transfer.get(idx=0, suffix=current_suffix)

        # Host case, never fit-intercept
        # 根据全局 y 的梯度计算 w 对应的梯度，需要为 fore_gradient * x
        unilateral_gradient = self.compute_gradient(data_instances, fore_gradient, False, need_average=False)
        return unilateral_gradient

    def compute_gradient_procedure(self, data_instances, cipher, model_weights,
                                   optimizer,
                                   n_iter_, batch_index):
        """
        Linear model gradient procedure
        Step 1: get host forwards which differ from different algorithm
                For Logistic Regression: forwards = wx


        """
        current_suffix = (n_iter_, batch_index)

        # 计算 Host 本地模型前向传播的结果, y = wx + b
        self.forwards = self.compute_forwards(data_instances, model_weights)

        if self.use_async:
            unilateral_gradient = self._asynchronous_compute_gradient(data_instances,
                                                                      cipher,
                                                                      current_suffix)
        else:
            unilateral_gradient = self._centralized_compute_gradient(data_instances,
                                                                     cipher,
                                                                     current_suffix)

        if optimizer is not None:
            unilateral_gradient = optimizer.add_regular_to_grad(unilateral_gradient, model_weights)

        # 将加密梯度传递给 Arbiter 参与方进行解密
        optimized_gradient = self.update_gradient(unilateral_gradient, suffix=current_suffix)
        LOGGER.debug(f"Before return compute_gradient_procedure")
        return optimized_gradient

    def compute_sqn_forwards(self, data_instances, delta_s, cipher):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*∑(0.25 * wx - 0.5 * y) * x
        y = ∇2^F(w_t)s_t = g' * s = (1/N)*∑(0.25 * x * s) * x
        define forward_hess = ∑(0.25 * x * s)
        """
        sqn_forwards = data_instances.mapValues(
            lambda v: cipher.encrypt(fate_operator.vec_dot(v.features, delta_s.coef_) + delta_s.intercept_))
        # forward_sum = sqn_forwards.reduce(reduce_add)
        return sqn_forwards

    def compute_forward_hess(self, data_instances, delta_s, forward_hess):
        """
        To compute Hessian matrix, y, s are needed.
        g = (1/N)*∑(0.25 * wx - 0.5 * y) * x
        y = ∇2^F(w_t)s_t = g' * s = (1/N)*∑(0.25 * x * s) * x
        define forward_hess = (0.25 * x * s)
        """
        hess_vector = self.compute_gradient(data_instances,
                                            forward_hess,
                                            delta_s.fit_intercept)
        return np.array(hess_vector)

    def remote_host_forward(self, host_forward, suffix=tuple()):
        self.host_forward_transfer.remote(obj=host_forward, role=consts.GUEST, idx=0, suffix=suffix)

    def get_fore_gradient(self, suffix=tuple()):
        host_forward = self.fore_gradient_transfer.get(idx=0, suffix=suffix)
        return host_forward

    # 使用 Arbiter 执行了梯度的解密
    def update_gradient(self, unilateral_gradient, suffix=tuple()):
        # 将同态加密的梯度发送给 Arbiter
        self.unilateral_gradient_transfer.remote(unilateral_gradient, role=consts.ARBITER, idx=0, suffix=suffix)
        # 获取 Arbiter 解密后的梯度
        optimized_gradient = self.unilateral_optim_gradient_transfer.get(idx=0, suffix=suffix)
        return optimized_gradient


class Arbiter(HeteroGradientBase):
    def __init__(self):
        super().__init__()
        self.has_multiple_hosts = False

    def _register_gradient_sync(self, guest_gradient_transfer, host_gradient_transfer,
                                guest_optim_gradient_transfer, host_optim_gradient_transfer):
        self.guest_gradient_transfer = guest_gradient_transfer
        self.host_gradient_transfer = host_gradient_transfer
        self.guest_optim_gradient_transfer = guest_optim_gradient_transfer
        self.host_optim_gradient_transfer = host_optim_gradient_transfer

    def compute_gradient_procedure(self, cipher, optimizer, n_iter_, batch_index):
        """
        Compute gradients.
        Received local_gradients from guest and hosts. Merge and optimize, then separate and remote back.
        Parameters
        ----------
        cipher: Use for encryption

        optimizer: optimizer that get delta gradient of this iter

        n_iter_: int, current iter nums

        batch_index: int

        """
        current_suffix = (n_iter_, batch_index)

        # 获取 Host 与 Guest 服务对应梯度
        host_gradients, guest_gradient = self.get_local_gradient(current_suffix)

        if len(host_gradients) > 1:
            self.has_multiple_hosts = True

        host_gradients = [np.array(h) for h in host_gradients]
        guest_gradient = np.array(guest_gradient)

        size_list = [h_g.shape[0] for h_g in host_gradients]
        size_list.append(guest_gradient.shape[0])

        gradient = np.hstack((h for h in host_gradients))
        gradient = np.hstack((gradient, guest_gradient))

        # 对同态加密的梯度进行解密
        grad = np.array(cipher.decrypt_list(gradient))

        # LOGGER.debug("In arbiter compute_gradient_procedure, before apply grad: {}, size_list: {}".format(
        #     grad, size_list
        # ))

        delta_grad = optimizer.apply_gradients(grad)

        # LOGGER.debug("In arbiter compute_gradient_procedure, delta_grad: {}".format(
        #     delta_grad
        # ))
        # 根据之前保存的 shape 列表 size_list 分离梯度
        separate_optim_gradient = self.separate(delta_grad, size_list)
        # LOGGER.debug("In arbiter compute_gradient_procedure, separated gradient: {}".format(
        #     separate_optim_gradient
        # ))
        host_optim_gradients = separate_optim_gradient[: -1]
        guest_optim_gradient = separate_optim_gradient[-1]

        # 将 Host 与 Guest 参与方解密后的梯度发送给对应的参与方
        self.remote_local_gradient(host_optim_gradients, guest_optim_gradient, current_suffix)
        return delta_grad

    @staticmethod
    def separate(value, size_list):
        """
        Separate value in order to several set according size_list
        Parameters
        ----------
        value: list or ndarray, input data
        size_list: list, each set size

        Returns
        ----------
        list
            set after separate
        """
        separate_res = []
        cur = 0
        for size in size_list:
            separate_res.append(value[cur:cur + size])
            cur += size
        return separate_res

    def get_local_gradient(self, suffix=tuple()):
        host_gradients = self.host_gradient_transfer.get(idx=-1, suffix=suffix)
        LOGGER.info("Get host_gradient from Host")

        guest_gradient = self.guest_gradient_transfer.get(idx=0, suffix=suffix)
        LOGGER.info("Get guest_gradient from Guest")
        return host_gradients, guest_gradient

    def remote_local_gradient(self, host_optim_gradients, guest_optim_gradient, suffix=tuple()):
        for idx, host_optim_gradient in enumerate(host_optim_gradients):
            self.host_optim_gradient_transfer.remote(host_optim_gradient,
                                                     role=consts.HOST,
                                                     idx=idx,
                                                     suffix=suffix)

        self.guest_optim_gradient_transfer.remote(guest_optim_gradient,
                                                  role=consts.GUEST,
                                                  idx=0,
                                                  suffix=suffix)


def compute_gradient(self, data_instances, fore_gradient, fit_intercept, need_average=True):
        is_sparse = data_overview.is_sparse_data(data_instances)

        LOGGER.debug("Use apply partitions")
        feat_join_grad = data_instances.join(fore_gradient,
                                             lambda d, g: (d.features, g))
        f = functools.partial(self.__apply_cal_gradient,
                              fixed_point_encoder=self.fixed_point_encoder,
                              is_sparse=is_sparse)
        gradient_sum = feat_join_grad.applyPartitions(f)
        gradient_sum = gradient_sum.reduce(lambda x, y: x + y)
        if fit_intercept:
            # bias_grad = np.sum(fore_gradient)
            bias_grad = fore_gradient.reduce(lambda x, y: x + y)
            gradient_sum = np.append(gradient_sum, bias_grad)

        if need_average:
            gradient = gradient_sum / data_instances.count()
        else:
            gradient = gradient_sum

        return gradient
