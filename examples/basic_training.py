from nums.core.application_manager import instance as ns
from nums.core.models import LogisticRegression

# Make dataset.
rs = ns.random_state(1337)
X1 = rs.normal(loc=5.0, shape=(500, 1), block_shape=(100, 1))
y1 = ns.zeros(shape=(500,), block_shape=(100,), dtype=bool)
X2 = rs.normal(loc=10.0, shape=(500, 1), block_shape=(100, 1))
y2 = ns.ones(shape=(500,), block_shape=(100,), dtype=bool)
X = ns.concatenate([X1, X2], axis=0)
y = ns.concatenate([y1, y2], axis=0)

model = LogisticRegression(ns, opt="newton", opt_params={"tol": 1e-8, "max_iter": 1})
model.fit(X, y)
y_pred = model.predict(X) > 0.5
print("accuracy", (ns.sum(y == y_pred) / X.shape[0]).get())
