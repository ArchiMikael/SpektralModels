import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.transforms import LayerPreprocess

learning_rate = 1e-3
epochs = 100
patience = 3600
data = "cora"


dataset = Citation(data, normalize_x=True, transforms=[LayerPreprocess(GCNConv)])


def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)

weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)


model = GCN(n_labels=dataset.n_labels)
model.compile(
    optimizer=Adam(learning_rate),
    loss=CategoricalCrossentropy(reduction="sum"),
    weighted_metrics=["acc"],
)


loader_tr = SingleLoader(dataset, sample_weights=weights_tr)
loader_va = SingleLoader(dataset, sample_weights=weights_va)
loader_te = SingleLoader(dataset, sample_weights=weights_te)


model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)


print("Evaluating model.")
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))


model.save("sp_cora_gcn.keras")
print("Exported")
loaded_model = keras.models.load_model("sp_cora_gcn.keras")
print("Imported")


model.summary()
loaded_model.summary()


print("Testing initial model")
loss_main, acc_main = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_main, acc_main))
print("Testing loaded model")
loss_loaded, acc_loaded = loaded_model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_loaded, acc_loaded))