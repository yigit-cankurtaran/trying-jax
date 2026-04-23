import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import jit

x_jnp = jnp.linspace(0, 10, 1000)  # can use jax arrays and numpy arrays together
y_jnp = 2 * jnp.sin(x_jnp) * jnp.cos(x_jnp)
plt.plot(x_jnp, y_jnp)
# plt.show()

# arrays are immutable, we can update individual elements with specific syntax
to_change = jnp.arange(10)
print(f"the array to change is {to_change}")  # prints [0 1 2 3 4 5 6 7 8 9]
changed = to_change.at[8].set(90)
print(f"the array after change is {changed}")  # prints [ 0  1  2  3  4  5  6  7 90  9]

# arrays can be sharded across multiple devices
# useful for multiple GPU and TPUs
print(to_change.devices())
print(to_change.sharding)


# can use JIT compilation if array shapes are static and known
def norm(x):
    x = x - x.mean(0)
    return x / x.std(0)


norm_compiled = jit(norm)

arr_to_norm = jnp.array(np.random.randint(10, 10000, 10))
print(norm_compiled(arr_to_norm))
