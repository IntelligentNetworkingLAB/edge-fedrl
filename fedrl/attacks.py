"""Privacy attacks on shared prototypes.

Client-ID attack:      MLP classifies which client produced a prototype centroid.
Actuator inference:    MLP regresses the client's action scale c_i.

Prototypes are split by TIME (early rounds -> attacker train, late rounds ->
attacker test) to avoid near-duplicate leakage within a round.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler


def run_attacks(npz_path, train_frac=0.6, seed=0):
    d = np.load(npz_path)
    mu, client, scale, rnd = d["mu"], d["client"], d["scale"], d["round"]
    rounds = np.unique(rnd)
    cut = rounds[int(len(rounds) * train_frac) - 1]
    tr, te = rnd <= cut, rnd > cut
    if te.sum() == 0 or tr.sum() == 0:
        raise ValueError("Not enough rounds for a time-based split")

    sc = StandardScaler().fit(mu[tr])
    Xtr, Xte = sc.transform(mu[tr]), sc.transform(mu[te])

    clf = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=seed)
    clf.fit(Xtr, client[tr])
    acc = float((clf.predict(Xte) == client[te]).mean())

    reg = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=seed)
    reg.fit(Xtr, scale[tr])
    mae = float(np.abs(reg.predict(Xte) - scale[te]).mean())

    n_clients = len(np.unique(client))
    return {"client_id_accuracy": acc, "random_baseline": 1.0 / n_clients,
            "actuator_mae": mae,
            "n_train": int(tr.sum()), "n_test": int(te.sum())}
