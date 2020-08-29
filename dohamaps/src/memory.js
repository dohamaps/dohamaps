
import * as tf from "@tensorflow/tfjs-node";

export async function chain(ops, tensor)
{
    var prev = tensor.clone();
    for (let op of ops)
    {
        var next = await op(prev);
        tf.dispose(prev);
        prev = next;
    }
    return next;
}

export function chainSync(ops, tensor)
{
    var prev = tensor.clone();
    for (let op of ops)
    {
        var next = op(prev);
        tf.dispose(prev);
        prev = next;
    }
    return next;
}
