import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess


class SGCN:
    def __init__(self, K):
        self.K = K

    def __call__(self, graph):
        out = graph.a
        for _ in range(self.K - 1):
            out = out.dot(out)
        out.sort_indices()
        graph.a = out
        return graph


K = 2  # Propagation steps for SGCN
dataset = Citation("cora", transforms=[LayerPreprocess(GCNConv), SGCN(K)])
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te


l2_reg = 5e-6  # L2 regularization rate
learning_rate = 0.2  # Learning rate
epochs = 100  # Number of training epochs
patience = 3600  # Patience for early stopping
a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1


N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes


@keras.saving.register_keras_serializable()
class SimpleGCNNet(Model):
    def __init__(self):
        super().__init__()
        self.gcnconv = GCNConv(n_out, activation="softmax", kernel_regularizer=l2(l2_reg), use_bias=False)

    def call(self, inputs):
        x, a = inputs
        out = self.gcnconv([x, a])

        return out


model = SimpleGCNNet()
optimizer = Adam(learning_rate=learning_rate)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", weighted_metrics=["acc"]
)


loader_tr = SingleLoader(dataset, sample_weights=mask_tr)
loader_va = SingleLoader(dataset, sample_weights=mask_va)
loader_te = SingleLoader(dataset, sample_weights=mask_te)
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


model.save("sp_cora_simple_gc.keras")
print("Exported")
loaded_model = keras.models.load_model("sp_cora_simple_gc.keras")
print("Imported")


model.summary()
loaded_model.summary()


print("Testing initial model")
loss_main, acc_main = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_main, acc_main))
print("Testing loaded model")
loss_loaded, acc_loaded = loaded_model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_loaded, acc_loaded))