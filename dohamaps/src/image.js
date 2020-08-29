
import * as tf from "@tensorflow/tfjs-node";
import * as UPNG from "upng-js";
import * as assert from "assert";

import * as util from "./util";

export function imageToTensor(buffer, channels = 3)
{
    function tidy()
    {
        return tf.cast(tf.node.decodeImage(buffer, channels), "float32");
    }
    return tf.tidy("image.imageToTensor", tidy);
}

export async function tensorToImage(tensor, channels = 3)
{
    async function tensorToArrayBuffer(frame)
    {
        try { return util.toArrayBuffer(Buffer.from(await frame.data())); }
        catch (error) { console.log(error); }
    }
    try
    {
        assert.deepEqual(tensor.rank, 3);
        const width = tensor.shape[1];
        const height = tensor.shape[0];
        const numSplits = tensor.shape[2] / channels;
        const frameTensors = tf.split(tensor, numSplits, 2);
        const frames = await Promise.all(frameTensors.map(tensorToArrayBuffer));
        tf.dispose(frameTensors);
        const delays = Array(frames.length).fill(0);
        return util.toBuffer(UPNG.encode(frames, width, height, 0, delays));
    }
    catch (error) { console.log(error); }
}

export function crop(image, top, left, dimensions)
{
    const height = dimensions[0];
    const width = dimensions[1];
    assert.deepEqual(image.rank, 3);
    const depth = image.shape[2];
    const begin = [ top, left, 0 ];
    const size = [ height, width, depth ];
    return tf.slice(image, begin, size);
}

export function cropRandom(image, dimensions)
{
    const cy = util.random(0, image.shape[0] - dimensions[0]);
    const cx = util.random(0, image.shape[1] - dimensions[1]);
    return crop(image, cy, cx, dimensions);
}

export function resize(image, scaleIndex, numScales)
{
    const rank = image.rank;
    const height = util.scale(image.shape[rank - 3], scaleIndex, numScales);
    const width = util.scale(image.shape[rank - 2], scaleIndex, numScales);
    return tf.image.resizeBilinear(image, [ height, width ]);
}

export function normalize(tensor)
{
    function tidy()
    {
        return tf.sub(tf.div(tensor, tf.scalar(127.5)), tf.scalar(1.0));
    }
    return tf.tidy("image.normalize", tidy);
}

export function denormalize(tensor)
{
    function tidy()
    {
        return tf.mul(tf.add(tensor, tf.scalar(1.0)), tf.scalar(127.5));
    }
    return tf.tidy("image.denormalize", tidy);
}

export function imageGradients(tensor)
{
    function tidy()
    {
        assert.deepEqual(tensor.rank, 4);
        const batchSize = tensor.shape[0];
        const height = tensor.shape[1];
        const width = tensor.shape[2];
        const channels = tensor.shape[3];

        const beginHeight = [ 0, 1, 0, 0 ];
        const beginWidth = [ 0, 0, 1, 0 ];
        const sizeHeight = [ batchSize, height - 1, width, channels ];
        const sizeWidth = [ batchSize, height, width - 1, channels ];
        const begin = [ 0, 0, 0, 0 ];

        var dy = tf.sub(tf.slice(tensor, beginHeight, sizeHeight), tf.slice(tensor, begin, sizeHeight));
        var dx = tf.sub(tf.slice(tensor, beginWidth, sizeWidth), tf.slice(tensor, begin, sizeWidth));

        dy = tf.concat4d([ dy, tf.zeros([ batchSize, 1, width, channels ], tensor.dtype) ], 1);
        dy = tf.reshape(dy, tensor.shape);

        dx = tf.concat4d([ dx, tf.zeros([ batchSize, height, 1, channels ], tensor.dtype) ], 2);
        dx = tf.reshape(dx, tensor.shape);

        return [ dy, dx ];
    }
    return tf.tidy("image.imageGradients", tidy);
}

export function psnr(x, y, maxVal)
{
    function tidy()
    {
        function log10(tensor)
        {
            return tf.div(tf.log(tensor), tf.log(tf.scalar(10.0)));
        }
        const mse = tf.mean(tf.squaredDifference(x, y), [ 1, 2, 3 ]);
        const a = tf.mul(tf.scalar(20), log10(tf.scalar(maxVal)));
        const b = tf.mul(tf.scalar(10), log10(mse));
        return tf.sub(a, b);
    }
    return tf.tidy("image.psnr", tidy);
}
