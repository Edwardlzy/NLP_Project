from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops

class LargebatchOptimizer(optimizer.Optimizer):
  '''Tensorflow implementation of the large batch training wrapper.
    
  '''

  def __init__(self, optimizer, update_steps, use_locking=False, name="Largebatch"):
    super(LargebatchOptimizer, self).__init__(use_locking, name)
    self.optimizer = optimizer
    self._update_steps = update_steps
    self._update_step = 0
    # self._la_step = 0
    # self._la_alpha = la_alpha
    # self._total_la_steps = la_steps

  def _create_slots(self, var_list):
    self.optimizer._create_slots(var_list)
    self._var_list = var_list

    first_var = min(var_list, key=lambda x: x.name)
    self._create_non_slot_variable(initial_value=self._update_step,
                                   name="update_step",
                                   colocate_with=first_var)
    
    # Create slots for the accumulated parameter gradients.
    for v in var_list:
      self._zeros_slot(v, "accumulated_grad", self._name)
      self._zeros_slot(v, "zero_grad", self._name)

  def _prepare(self):
    self.optimizer._prepare()

    update_steps = self._call_if_callable(self._update_steps)
    self._update_steps_t = ops.convert_to_tensor(update_steps, name="update_steps")

    # la_alpha = self._call_if_callable(self._la_alpha)
    # total_la_steps = self._call_if_callable(self._total_la_steps)
    
    # self._la_alpha_t = ops.convert_to_tensor(la_alpha, name="la_alpha")
    # self._total_la_steps_t = ops.convert_to_tensor(total_la_steps, name="total_la_steps")


  def _get_update_step_accumulators(self):
    with ops.init_scope():
      if context.executing_eagerly():
        graph = None
      else:
        graph = ops.get_default_graph()
      return self._get_non_slot_variable("update_step", graph=graph)

  def process_grad(self, grad):
    update_step = self._get_update_step_accumulators()
    with ops.colocate_with(update_step):
      def accumulate_grad():
        # Add current gradients to the cached gradients.
        ag = [self.get_slot(cur_g, "accumulated_grad").assign(self.get_slot(cur_g, "accumulated_grad") + cur_g) for cur_g in grad]
        control_flow_ops.group([update_step.assign(update_step + 1, use_locking=self._use_locking)] + ag)
        return [self.get_slot(cur_g, "zero_grad") for cur_g in grad]
      def update_grad():
        # Average the gradients and reset the update_step.
        avg_grads = [self.get_slot(cur_g, "accumulated_grad") / update_step for cur_g in grad]
        # avg_grads_and_vars = avg_grads_and_vars, grads_and_vars[1]
        # apply_grad = self.optimizer.apply_gradients(avg_grads_and_vars)

        update_step = update_step.assign(1, use_locking=self._use_locking)
        new_grad = [self.get_slot(cur_g, "accumulated_grad").assign(cur_g) for cur_g in grad]
        control_flow_ops.group(avg_grads + [update_step] + new_grad)
        return avg_grads
      condition = tf.greater_equal(update_step, self._update_steps_t)
      update_params_states = tf.cond( condition,
                                      update_grad,
                                      accumulate_grad,
                                    )
    return control_flow_ops.group([update_params_states])


  def _apply_dense(self, grad, var):
    grad = self.process_grad(grad)
    return self.optimizer._apply_dense(grad, var)

  def _resource_apply_dense(self, grad, var):
    grad = self.process_grad(grad)
    return self.optimizer._resource_apply_dense(grad, var)

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
    grad = self.process_grad(grad)
    return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

  def _apply_sparse(self, grad, var):
    grad = self.process_grad(grad)
    return self.optimizer._apply_sparse(grad, var)

  def _resource_scatter_add(self, x, i, v):
    return self.optimizer._resource_scatter_add(x, i, v)
    
  def _resource_apply_sparse(self, grad, var, indices):
    grad = self.process_grad(grad)
    return self.optimizer._resource_apply_sparse(grad, var, indices)

  # def minimize(self, loss, global_step=None, var_list=None,
  #              gate_gradients=1, aggregation_method=None,
  #              colocate_gradients_with_ops=False, name=None,
  #              grad_loss=None):

  #   grads_and_vars = self.optimizer.compute_gradients(
  #       loss, var_list=var_list, gate_gradients=gate_gradients,
  #       aggregation_method=aggregation_method,
  #       colocate_gradients_with_ops=colocate_gradients_with_ops,
  #       grad_loss=grad_loss)

  #   vars_with_grad = [v for g, v in grads_and_vars if g is not None]
  #   if not vars_with_grad:
  #     raise ValueError(
  #         "No gradients provided for any variable, check your graph for ops"
  #         " that do not support gradients, between variables %s and loss %s." %
  #         ([str(v) for _, v in grads_and_vars], loss))

  #   with ops.control_dependencies([grads_and_vars,]):
  #     update_step = self._get_update_step_accumulators()
  #     with ops.colocate_with(update_step):
  #       def accumulate_grad():
  #         # Add current gradients to the cached gradients.
  #         ag = [self.get_slot(cur_g, "accumulated_grad").assign(self.get_slot(cur_g, "accumulated_grad") + cur_g) for cur_g, _ in grads_and_vars]
  #         return control_flow_ops.group([update_step.assign(update_step + 1, use_locking=self._use_locking)] + ag)
  #       def update_grad():
  #         # Average the gradients and reset the update_step.
  #         avg_grads_and_vars = [self.get_slot(cur_g, "accumulated_grad") / update_step for cur_g, _ in grads_and_vars]
  #         avg_grads_and_vars = avg_grads_and_vars, grads_and_vars[1]
  #         apply_grad = self.optimizer.apply_gradients(avg_grads_and_vars)

  #         update_step = update_step.assign(1, use_locking=self._use_locking)
  #         new_grad = [self.get_slot(cur_g, "accumulated_grad").assign(cur_g) for cur_g, _ in grads_and_vars]

  #         return control_flow_ops.group(avg_grads_and_vars + [apply_grad, update_step] + new_grad)

  #       condition = tf.greater_equal(update_step, self._update_steps_t)
  #       update_params_states = tf.cond( condition,
  #                                       update_grad,
  #                                       accumulate_grad,
  #                                     )
  #   return control_flow_ops.group([update_params_states])

  def _finish(self, update_ops, name_scope):
    update_step = self._get_update_step_accumulators()
    return control_flow_ops.group([self.optimizer._finish(update_ops, name_scope)], name=name_scope)

  # def _finish(self, update_ops, name_scope):
  #   inner_finish_op = self.optimizer._finish(update_ops, name_scope)

  #   with ops.control_dependencies([inner_finish_op,]):
  #     update_step = self._get_update_step_accumulators()
  #     with ops.colocate_with(update_step):  # Why?

  #       def update_update_step_func():
  #         ## update the update_step
  #         return control_flow_ops.group([ update_step.assign(
  #             update_step + 1, use_locking=self._use_locking),])

  #       def pull_back_func():
  #         ## update the update_step
  #         update_update_step = update_step.assign(
  #             0, use_locking=self._use_locking)
  #         ## interpolate the variables
  #         interpolation = [v.assign(self.get_slot(v, "cached_params") + self._la_alpha_t*(v - self.get_slot(v, "cached_params"))) for v in self._var_list]
          
  #         ## update the cached params
  #         with ops.control_dependencies(interpolation):
  #           update_cached_params = [self.get_slot(v, "cached_params").assign(updated_v) for v, updated_v in zip(self._var_list, interpolation)]
  #         return control_flow_ops.group([update_update_step,] +  interpolation + update_cached_params)

  #       ## condition for when to pull back the params
  #       condition = tf.greater_equal(la_step, self._total_la_steps_t)
  #       update_lookahead_states = tf.cond( condition,
  #                                pull_back_func,
  #                                update_la_step_func,
  #                               )

  #   return control_flow_ops.group([inner_finish_op, update_lookahead_states],
  #                                 name=name_scope)
    ## Update the power accumulators.
    #with ops.control_dependencies(update_ops):
    #  beta1_power, beta2_power = self._get_beta_accumulators()
    #  with ops.colocate_with(beta1_power):
    #    update_beta1 = beta1_power.assign(
    #        beta1_power * self._beta1_t, use_locking=self._use_locking)
    #    update_beta2 = beta2_power.assign(
    #        beta2_power * self._beta2_t, use_locking=self._use_locking)
    #return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
    #                              name=name_scope)


  def _call_if_callable(self, param):
    """Call the function if param is callable."""
    return param() if callable(param) else param

  #def _apply_sparse(self, grad, noise, var):
  #  lr = (self._lr_t *
  #        math_ops.sqrt(1 - self._beta2_power)
  #        / (1 - self._beta1_power))
  #  # m_t = beta1 * m + (1 - beta1) * g_t
  #  m = self.get_slot(var, "m")
  #  m_scaled_g_values = grad * (1 - self._beta1_t)
  #  m_t = state_ops.assign(m, m * self._beta1_t,
  #                         use_locking=self._use_locking)
  #  m_t = state_ops.assign_add(m_t, m_scaled_g_values,
  #                             use_locking=self._use_locking)
  #  # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
  #  v = self.get_slot(var, "v")
  #  v_scaled_g_values = (grad * grad) * (1 - self._beta2_t)
  #  v_t = state_ops.assign(v, v * self._beta2_t, use_locking=self._use_locking)
  #  v_t = state_ops.assign_add(v_t, v_scaled_g_values,
  #                             use_locking=self._use_locking)
  #  v_sqrt = math_ops.sqrt(v_t)
  #  var_update = state_ops.assign_sub(var,
  #                                    lr * m_t / (v_sqrt + self._epsilon_t) - noise,
  #                                    use_locking=self._use_locking)
  #  return control_flow_ops.group(*[var_update, m_t, v_t])

  #def apply_gradients(self, grads_and_vars, noise, global_step=None, name=None):
  #  # This is a default implementation of apply_gradients() that can be shared
  #  # by most optimizers.  It relies on the subclass implementing the following
  #  # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().
  #  grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works
  #  for g, v in grads_and_vars:
  #    if not isinstance(g, (ops.Tensor, ops.IndexedSlices, type(None))):
  #      raise TypeError(
  #          "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
  #    if not isinstance(v, variables.Variable):
  #      raise TypeError(
  #          "Variable must be a tf.Variable: %s" % v)
  #    if g is not None:
  #      self._assert_valid_dtypes([g, v])
  #  var_list = [v for g, v in grads_and_vars if g is not None]
  #  if not var_list:
  #    raise ValueError("No gradients provided for any variable: %s" %
  #                     (grads_and_vars,))
  #  with ops.control_dependencies(None):
  #    self._create_slots(var_list)
  #  update_ops = []
  #  with ops.op_scope([], name, self._name) as name:
  #    self._prepare()
  #    for (grad, var), n in zip(grads_and_vars, noise):
  #      if grad is None:
  #        continue
  #      with ops.name_scope("update_" + var.op.name), ops.device(var.device):
  #        update_ops.append(self._apply_sparse(grad, n, var))
  #    if global_step is None:
  #      return self._finish(update_ops, name)
  #    else:
  #      with ops.control_dependencies([self._finish(update_ops, "update")]):
  #        with ops.colocate_with(global_step):
  #          return state_ops.assign_add(global_step, 1, name=name).op

