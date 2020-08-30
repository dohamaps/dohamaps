
import * as tf from "@tensorflow/tfjs-node";
import * as assert from "assert";

import * as image from "./image";

export function adversarial(yTrue, yPred)
{
    function tidy() { return tf.metrics.binaryCrossentropy(yTrue, yPred); }
    return tf.tidy("losses.adversarial", tidy);
}

export function gdl(yTrue, yPred, c = 2)
{
    function tidy()
    {
        const yTrueD = image.imageGradients(yTrue);
        const yPredD = image.imageGradients(yPred);

        const gDiffY = tf.abs(tf.sub(tf.abs(yTrueD[0]), tf.abs(yPredD[0])));
        const gDiffX = tf.abs(tf.sub(tf.abs(yTrueD[1]), tf.abs(yPredD[1])));

        const powX = tf.pow(gDiffX, c);
        const powY = tf.pow(gDiffY, c);
        return tf.mean(tf.add(powX, powY));
    }
    return tf.tidy("losses.gdl", tidy);
}


export function lp(yTrue, yPred, lNum = 2)
{
    function tidy() { return tf.sum(tf.abs(yPred - yTrue) ** lNum); }
    return tf.tidy("losses.lp", tidy);
}


export function combined(yTrue, yPred, labels, alpha = 0.05, beta = 1, gamma = 1, lNum = 2)
{
    function tidy()
    {
        console.log(labels);
        const batchSize = yPred.shape[0];

        var loss = tf.mul(tf.scalar(beta), lp(yTrue, yPred, lNum));
        loss = tf.add(loss, tf.mul(tf.scalar(gamma), gdl(yTrue, yPred)));
        const ones = tf.ones([ batchSize, 1 ]);
        loss = tf.add(loss, tf.mul(tf.scalar(alpha), adversarial(ones, labels)));

        return loss;
    }
    return tf.tidy("losses.combined", tidy);
}
