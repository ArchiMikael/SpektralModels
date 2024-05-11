import numpy as np
from tensorflow import keras
from keras.models import Model

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess


data = "cora"
dataset = Citation(data, normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)
weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)
loader_te = SingleLoader(dataset, sample_weights=weights_te)


print("----------Importing----------")
loaded_model = keras.models.load_model("sp_cora_gcn.keras")
loaded_model.summary()
print("----------Imported----------")


print("Testing loaded model")
loss_loaded, acc_loaded = loaded_model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_loaded, acc_loaded))