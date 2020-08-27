
import * as tf from "@tensorflow/tfjs-node";
import * as UPNG from "upng-js";
import * as assert from "assert";

import * as util from "./util";

export function imageToTensor(imageBuffer)
{
    return tf.cast(tf.node.decodeImage(imageBuffer), "float32");
}

export async function tensorToImage(tensor, channels)
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
        var frames = tf.split(tensor, numSplits, 2);
        frames = await Promise.all(frames.map(tensorToArrayBuffer));
        const delays = Array(frames.length).fill(0);
        return util.toBuffer(UPNG.encode(frames, width, height, 0, delays));
    }
    catch (error) { console.log(error); }
}

export function crop(image, top, left, height, width)
{
    assert.deepEqual(image.rank, 3);
    const depth = image.shape[2];
    const begin = [ top, left, 0 ];
    const size = [ height, width, depth ];
    return tf.slice(image, begin, size);
}

export function cropRandom(image, height, width)
{
    const cy = util.random(0, image.shape[0] - height);
    const cx = util.random(0, image.shape[1] - width);
    return crop(image, cy, cx, height, width);
}

export function imageGradients(tensor)
{
    assert.deepEqual(tensor.rank, 4);
    const batchSize = tensor.shape[0];
    const height = tensor.shape[1];
    const width = tensor.shape[2];
    const channels = tensor.shape[3];

    const beginY = [ 0, 1, 0, 0 ];
    const beginX = [ 0, 0, 1, 0 ];
    const endY = [ batchSize, height - 1, width, channels ];
    const endX = [ batchSize, height, width - 1, channels ];
    const begin = [ 0, 0, 0, 0 ];
    const end = [ batchSize, height, width, channels ];

    var dy = tf.sub(tf.slice(tensor, beginY, end), tf.slice(tensor, begin, endY));
    var dx = tf.sub(tf.slice(tensor, beginX, end), tf.slice(tensor, begin, endX));

    dy = tf.concat([ dy, tf.zeros([ batchSize, height, 1, channels ], tensor.dtype) ], 1);
    dy = tf.reshape(dy, tensor.shape);

    dx = tf.concat([ dx, tf.zeros([ batchSize, 1, width, channels ], tensor.dtype) ], 2);
    dx = tf.reshape(dx, tensor.shape);

    return [ dy, dx ];
}
