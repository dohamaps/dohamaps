
import * as tf from "@tensorflow/tfjs-node";
import * as assert from "assert";

import * as image from ".image";

export function adversarial(yTrue, yPred)
{
    assert.deepEqual(yTrue.length, yPred.length);
    const losses = [  ];
    for (let i = 0; i < yPred.length; ++i)
        losses.push(tf.metrics.binaryCrossentropy(yTrue[i], yPred[i]));

    return tf.mean(tf.stack(losses));
}


export function gdl(yTrue, yPred, c)
{
    assert.deepEqual(yTrue.length, yPred.length);
    const losses = [];
    for (let i = 0; i < yPred.length; ++i)
        const yTrueD = tf.image.imageGradients(yTrue[i]);
        const yPredD = tf.image.imageGradients(yPred[i]);

        const gDiffY = tf.abs(tf.abs(yTrueD[0]) - tf.abs(yPredD[0]));
        const gDiffX = tf.abs(tf.abs(yTrueD[1]) - tf.abs(yPredD[1]));

        const powX = tf.pow(gDiffX, c);
        const powY = tf.pow(gDiffY, c);

        losses.push(tf.mean(powX + powY));

    return losses;
}


function lp(yTrue, yPred, lNum)
{
    assert.deepEqual(yTrue.length, yPred.length);
    const losses = [  ];
    for (let i = 0; i < yPred.length; ++i)
        losses.push(tf.sum(tf.abs(yPred[i] - yTrue[i]) ** lNum));

    return tf.mean(tf.stack(losses));
}


function combined(yTrue, yPred, labels, alpha, beta, gamma, lNum)
{
    assert.deepEqual(yTrue.length, labels.length);
    assert.deepEqual(yPred.length, labels.length);
    const batchSize = yPred[0].shape[0];

    var loss = tf.mul(tf.scalar(beta), lp(yTrue, yPred, lNum));
    loss = tf.add(loss, tf.mul(tf.scalar(gamma), gdl(yTrue, yPred)));
    ones = [  ];
    for (let i = 0; i < labels.length; ++i)
        ones.push(tf.ones([ batchSize, 1 ]));
    loss = tf.add(loss, tf.mul(tf.scalar(alpha), adversarial(ones, labels)));

    return loss;
}
