
import * as tf from "@tensorflow/tfjs-node";

import * as image from "./image";

function psnrFunction(yTrue, yPred)
{
    function tidy()
    {
        return tf.mean(image.psnr(yTrue, yPred, 2.0));
    }
    return tf.tidy("metrics.psnr", tidy);
}

function sharpdiffFunction(yTrue, yPred)
{
    function tidy()
    {
        function log10(tensor)
        {
            return tf.div(tf.log(tensor), tf.log(tf.scalar(10.0)));
        }

        const shape = yPred.shape;
        var numPixelsInv = tf.scalar(shape[1] * shape[2] * shape[3], "float32");
        numPixelsInv = tf.div(tf.scalar(1.0), numPixelsInv);

        const yTrueD = image.imageGradients(yTrue);
        const yPredD = image.imageGradients(yPred);

        const predGradSum = tf.add(yPredD[1], yPredD[0]);
        const trueGradSum = tf.add(yTrueD[1], yTrueD[0]);

        const gradDiff = tf.abs(tf.sub(trueGradSum, predGradSum));
        const gradDiffRed = tf.sum(gradDiff, [ 1, 2, 3 ]);

        const term = tf.div(tf.scalar(1.0), tf.mul(numPixelsInv, gradDiffRed));

        const batchErrors = tf.mul(tf.scalar(10.0), log10(term));
        return tf.mean(batchErrors);
    }
    return tf.tidy("metrics.sharpdiff", tidy);
}

export const psnr = { name: "psnr", apply: psnrFunction };
export const sharpdiff = { name: "sharpdiff", apply: sharpdiffFunction };
