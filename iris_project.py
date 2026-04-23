import jax
import jax.numpy as jnp
from jax import random, grad, jit
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# standard scaler for z-score normalization
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data.astype(jnp.float32)
y = iris.target.astype(jnp.int32)

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(jnp.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

X_train = jnp.array(X_train)
X_test = jnp.array(X_test)
y_train = jnp.array(y_train)
y_test = jnp.array(y_test)


def init_params(key, input_dim=4, hidden_dim=16, output_dim=3):
    # multiple keys
    k1, k2 = random.split(key)
