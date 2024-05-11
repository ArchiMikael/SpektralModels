import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from tensorflow.random import set_seed

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess


set_seed(0)


def mask_to_weights(mask):
    return mask.astype(np.float32) / np.count_nonzero(mask)


data = "cora"
dataset = Citation(data, normalize_x=True, transforms=[LayerPreprocess(GATConv)])
weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)


channels = 8  # Number of channels in each head of the first GAT layer
n_attn_heads = 8  # Number of attention heads in first GAT layer
dropout = 0.6  # Dropout rate for the features and adjacency matrix
l2_reg = 2.5e-4  # L2 regularization rate
learning_rate = 0.01  # Learning rate
epochs = 100  # Number of training epochs
patience = 3600  # Patience for early stopping


N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes


@keras.saving.register_keras_serializable()
class GATNet(Model):
    def __init__(self):
        super().__init__()
        self.dropout1 = Dropout(dropout)
        self.gatconv1 = GATConv(
          channels,
          attn_heads=n_attn_heads,
          concat_heads=True,
          dropout_rate=dropout,
          activation="elu",
          kernel_regularizer=l2(l2_reg),
          attn_kernel_regularizer=l2(l2_reg),
          bias_regularizer=l2(l2_reg),
        )
        self.dropout2 = Dropout(dropout)
        self.gatconv2 = GATConv(
          n_out,
          attn_heads=1,
          concat_heads=False,
          dropout_rate=dropout,
          activation="softmax",
          kernel_regularizer=l2(l2_reg),
          attn_kernel_regularizer=l2(l2_reg),
          bias_regularizer=l2(l2_reg),
        )

    def call(self, inputs):
        x, a = inputs
        do_1 = self.dropout1(x)
        gc_1 = self.gatconv1([do_1, a])
        do_2 = self.dropout1(gc_1)
        gc_2 = self.gatconv2([do_2, a])

        return gc_2


model = GATNet()
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer,
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


model.save("sp_cora_gat.keras")
print("Exported")
loaded_model = keras.models.load_model("sp_cora_gat.keras")
print("Imported")


model.summary()
loaded_model.summary()


print("Testing initial model")
loss_main, acc_main = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_main, acc_main))
print("Testing loaded model")
loss_loaded, acc_loaded = loaded_model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_loaded, acc_loaded))