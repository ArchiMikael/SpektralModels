from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import ARMAConv
from spektral.transforms import LayerPreprocess


dataset = Citation("cora", transforms=[LayerPreprocess(ARMAConv)])
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te


channels = 16  # Number of channels in the first layer
iterations = 1  # Number of iterations to approximate each ARMA(1)
order = 2  # Order of the ARMA filter (number of parallel stacks)
share_weights = True  # Share weights in each ARMA stack
dropout_skip = 0.75  # Dropout rate for the internal skip connection of ARMA
dropout = 0.5  # Dropout rate for the features
l2_reg = 5e-4  # L2 regularization rate
learning_rate = 1e-4  # Learning rate
epochs = 10000  # Number of training epochs
patience = 3600  # Patience for early stopping
a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1


N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes


@keras.saving.register_keras_serializable()
class ARMANet(Model):
    def __init__(self):
        super().__init__()
        self.armaconv1 = ARMAConv(
          channels,
          iterations=iterations,
          order=order,
          share_weights=share_weights,
          dropout_rate=dropout_skip,
          activation="elu",
          gcn_activation="elu",
          kernel_regularizer=l2(l2_reg),
        )
        self.dropout = Dropout(dropout)
        self.armaconv2 = ARMAConv(
          n_out,
          iterations=1,
          order=1,
          share_weights=share_weights,
          dropout_rate=dropout_skip,
          activation="softmax",
          gcn_activation=None,
          kernel_regularizer=l2(l2_reg),
        )

    def call(self, inputs):
        x, a = inputs
        gc_1 = self.armaconv1([x, a])
        gc_2 = self.dropout(gc_1)
        gc_2 = self.armaconv2([gc_2, a])

        return gc_2


model = ARMANet()
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


model.save("sp_cora_arma.keras")
print("Exported")
loaded_model = keras.models.load_model("sp_cora_arma.keras")
print("Imported")


model.summary()
loaded_model.summary()


print("Testing initial model")
loss_main, acc_main = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_main, acc_main))
print("Testing loaded model")
loss_loaded, acc_loaded = loaded_model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_loaded, acc_loaded))