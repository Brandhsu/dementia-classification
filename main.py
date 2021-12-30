import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, layers, metrics, losses, optimizers

from jarvis.train import params
from jarvis.utils.general import gpus, overload

from tfcaidm import JClient
from tfcaidm import Model
from tfcaidm import Trainer
from tfcaidm.models import head, registry

import custom_losses as custom


# --- Data
@overload(JClient)
def create_generator(self, gen_data):
    """Modifies dataset generator (applies same modification to train and valid generators)

    Each training pass will consist of N-1 contrastive comparisons
    Note using custom layers so inputs and outputs are stored in x
    """

    for x, y in gen_data:

        # --- Prepare ground-truths
        xs = x["dat"]
        ys = x["lbl"]

        if xs.shape[0] > 1 and ys.shape[0] > 1:
            xs_unk = xs[:-1]
            ys_unk = ys[:-1]
            xs_anc = np.stack([xs[-1]] * len(xs_unk))
            ys_anc = np.stack([ys[-1]] * len(ys_unk))
        else:
            xs_unk = xs
            ys_unk = ys
            xs_anc = xs
            ys_anc = ys

        # --- Assign ground-truths
        xs = {}
        ys = {}

        xs["anc"] = xs_anc
        xs["unk"] = xs_unk
        ys["ctr"] = tf.cast((ys_anc == ys_unk), tf.float32)
        ys["euc"] = tf.cast((ys_anc == ys_unk), tf.float32)
        ys["ned"] = tf.cast((ys_anc == ys_unk), tf.float32)
        ys["cls_anc"] = ys_anc
        ys["cls_unk"] = ys_unk

        yield xs, ys


# --- Model
def AE(INPUT_SHAPE, hyperparams):
    autoencoder = registry.available_models()["ae"]

    # --- Extract hyperparams
    n = hyperparams["model"]["depth"]
    c = hyperparams["model"]["width"]
    k = hyperparams["model"]["kernel_size"]
    s = hyperparams["model"]["strides"]
    e = hyperparams["model"]["embed"]

    features = autoencoder(INPUT_SHAPE, n, c, k, s, hyperparams)
    embed = head.Encoder.last_layer(**features)

    # --- NOTE: The below hyperparameters will not be logged!
    conv = lambda filters, name: layers.Conv3D(
        filters=filters,
        kernel_size=1,
        activation="sigmoid",
        name=name,
        padding="same",
    )

    ftr = layers.GlobalAveragePooling3D()(embed)
    ftr = layers.Reshape((1, 1, 1, embed.shape[-1]))(ftr)
    ctr = conv(filters=e, name="ctr")(ftr)
    cls = conv(filters=1, name="cls")(ctr)

    logits = {}
    logits["ctr"] = ctr
    logits["cls"] = cls

    return logits


@overload(Model)
def create(self):

    # --- User defined code
    INPUT_SHAPE = (96, 160, 160, 1)
    inputs = Input(shape=INPUT_SHAPE)
    outputs = AE(inputs, self.hyperparams)

    # --- Create tensorflow model
    backbone = self.assemble(inputs=inputs, outputs=outputs)

    inputs = {
        "anc": Input(shape=INPUT_SHAPE, name="anc"),
        "unk": Input(shape=INPUT_SHAPE, name="unk"),
    }

    # --- Define contrastive network
    anc_net = backbone(inputs=inputs["anc"])
    unk_net = backbone(inputs=inputs["unk"])

    # --- Eembeddings
    ctr = layers.Lambda(custom.cosine_similarity)([anc_net["ctr"], unk_net["ctr"]])
    euc = layers.Lambda(custom.euclidean_distance)([anc_net["ctr"], unk_net["ctr"]])
    ned = layers.Lambda(custom.norm_euclidean_distance)(
        [anc_net["ctr"], unk_net["ctr"]]
    )

    logits = {}
    logits["ctr"] = layers.Layer(name="ctr")(ctr)
    logits["euc"] = layers.Layer(name="euc")(euc)
    logits["ned"] = layers.Layer(name="ned")(ned)
    logits["cls_anc"] = layers.Layer(name="cls_anc")(anc_net["cls"])
    logits["cls_unk"] = layers.Layer(name="cls_unk")(unk_net["cls"])

    # --- Create tensorflow model
    model = self.assemble(inputs=inputs, outputs=logits)

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=self.hyperparams["train"]["trainer"]["lr"]
        ),
        loss={
            "ctr": custom.ContrastiveLoss(gamma=self.hyperparams["model"]["gamma"])
            if self.hyperparams["model"]["log_loss"] == False
            else custom.LogContrastiveLoss(gamma=self.hyperparams["model"]["gamma"]),
            "euc": custom.ContrastiveLoss(gamma=self.hyperparams["model"]["gamma"])
            if self.hyperparams["model"]["log_loss"] == False
            else custom.LogContrastiveLoss(gamma=self.hyperparams["model"]["gamma"]),
            "ned": custom.ContrastiveLoss(gamma=self.hyperparams["model"]["gamma"])
            if self.hyperparams["model"]["log_loss"] == False
            else custom.LogContrastiveLoss(gamma=self.hyperparams["model"]["gamma"]),
            "cls_anc": losses.BinaryCrossentropy(),
            "cls_unk": losses.BinaryCrossentropy(),
        },
        loss_weights={
            "ctr": self.hyperparams["model"]["loss_weights"]["ctr"],
            "euc": self.hyperparams["model"]["loss_weights"]["euc"],
            "ned": self.hyperparams["model"]["loss_weights"]["ned"],
            "cls_anc": self.hyperparams["model"]["loss_weights"]["cls_anc"],
            "cls_unk": self.hyperparams["model"]["loss_weights"]["cls_unk"],
        },
        metrics={
            "cls_anc": metrics.BinaryAccuracy(),
            "cls_unk": metrics.BinaryAccuracy(),
        },
        experimental_run_tf_function=False,
    )

    return model


if __name__ == "__main__":

    # --- Setup
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    gpus.autoselect()
    hyperparams = params.load()

    # --- Train model (dataset and model created within trainer)
    trainer = Trainer(hyperparams)
    results = trainer.cross_validation(save=True)
    trainer.save_results(results)
