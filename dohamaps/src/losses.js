
import * as tf from "@tensorflow/tfjs-node";
import * as assert from "assert";

import * as image from "./image";

export function adversarial(yTrue, yPred)
{
    function tidy()
    {
        assert.deepEqual(yTrue.length, yPred.length);
        const losses = [  ];
        for (let i = 0; i < yPred.length; ++i)
            losses.push(tf.metrics.binaryCrossentropy(yTrue[i], yPred[i]));

        return tf.mean(tf.stack(losses));
    }
    return tf.tidy("losses.adversarial", tidy);
}


export function gdl(yTrue, yPred, c = 2)
{
    function tidy()
    {
        assert.deepEqual(yTrue.length, yPred.length);
        const losses = [];
        for (let i = 0; i < yPred.length; ++i)
        {
            const yTrueD = image.imageGradients(yTrue[i]);
            const yPredD = image.imageGradients(yPred[i]);

            const gDiffY = tf.abs(tf.sub(tf.abs(yTrueD[0]), tf.abs(yPredD[0])));
            const gDiffX = tf.abs(tf.sub(tf.abs(yTrueD[1]), tf.abs(yPredD[1])));

            const powX = tf.pow(gDiffX, c);
            const powY = tf.pow(gDiffY, c);

            losses.push(tf.mean(tf.add(powX, powY)));
        }
        return tf.mean(tf.stack(losses));
    }
    return tf.tidy("losses.gdl", tidy);
}


export function lp(yTrue, yPred, lNum = 2)
{
    function tidy()
    {
        assert.deepEqual(yTrue.length, yPred.length);
        const losses = [  ];
        for (let i = 0; i < yPred.length; ++i)
            losses.push(tf.sum(tf.abs(yPred[i] - yTrue[i]) ** lNum));

        return tf.mean(tf.stack(losses));
    }
    return tf.tidy("losses.lp", tidy);
}


export function combined(yTrue, yPred, labels, alpha = 0.05, beta = 1, gamma = 1, lNum = 2)
{
    function tidy()
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
    return tf.tidy("losses.combined", tidy);
}
