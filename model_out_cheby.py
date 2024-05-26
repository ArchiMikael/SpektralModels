from cheby_net import *


print("----------Importing----------")
loaded_model = keras.models.load_model("sp_cora_cheby.keras")
loaded_model.summary()
print("----------Imported----------")


print("Testing loaded model")
loss_loaded, acc_loaded = loaded_model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done. Test loss: {}. Test acc: {}".format(loss_loaded, acc_loaded))