import jax
import optax
import jax.numpy as jnp

def scaling_sketch(mnmx, resolution):
    mn1, mx1, mn2, mx2 = mnmx
    gg1 = jnp.logspace(mn1, mx1, resolution)
    gg2 = jnp.logspace(mn2, mx2, resolution)
    lr0, lr1 = jnp.meshgrid(gg2, gg1)
    lr = jnp.stack([lr0.ravel(), lr1.ravel()], axis=-1)
    return lr

@jax.jit
def train_step(state, batch):
    """Train for a single step."""
    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x=batch['image'], on_train=True, mutable=['batch_stats'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
        return loss, (logits, updates)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return state, metrics

@jax.jit
def evaluate_step(state, batch):
    """Evaluate for a single step."""
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x=batch['image'], on_train=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
        return loss, logits
    loss, logits = loss_fn(state.params)
    metrics = {
        'loss': loss,
        'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label']),
    }
    return metrics


if __name__ == "__main__":
    import os
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

    from datasets.mnist import *
    from model.resnet_v4 import *
    from flax.training import train_state
    from typing import Any
    import jax.numpy as jnp
    import optax
    train_ds, test_ds = prepare_dataset(32)
    batch = next(iter(train_ds))
    batch['image'] = jnp.array(batch['image'], dtype=jnp.float32)
    batch['label'] = jnp.array(batch['label'], dtype=jnp.int8)

    # lowered HLO
    class TrainState(train_state.TrainState):
        batch_stats: Any

    model = ResNet(10, flax.linen.relu, ResNetBlock)
    variables = model.init(jax.random.PRNGKey(1), batch['image'])
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        tx=optax.sgd(learning_rate=0.001)
        )
    lowered = train_step.lower(state, batch)
    print(lowered.as_text())

    # compiled HLO
    compiled = lowered.compile()
    print(compiled.cost_analysis()[0]['flops'])
    
    # Run
    print(compiled(state, batch))

    # Compile-check evaluate_step, too
    lowered_eval = evaluate_step.lower(state, batch)
    compiled_eval = lowered_eval.compile()
    print(lowered_eval.as_text())
    print(compiled_eval.cost_analysis()[0]['flops'])
    
    # pmapping check
    print(jax.devices())
    @jax.pmap
    def create_first(rng):
        model = ResNet(10, flax.linen.relu, ResNetBlock)
        params = model.init(rng, batch['image'])['params']
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            batch_stats=variables['batch_stats'],
            tx=optax.sgd(learning_rate=0.001)
        )
    rng = jax.random.split(jax.random.PRNGKey(42), 4)
    p_state = create_first(rng)
    
    p_train = jax.pmap(
        train_step, 
        axis_name='i', 
        in_axes=(0, None),
        devices=jax.devices()
        )(p_state, batch)
    print(len(p_train))

    # 다시 병렬의 길이 열린듯