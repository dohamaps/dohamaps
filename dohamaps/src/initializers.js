
import * as tf from "@tensorflow/tfjs-node";

export function truncatedNormal()
{
    const config =
    {
        mean: 0.0,
        stddev: 0.01,
        seed: null
    };
    return tf.initializers.truncatedNormal(config);
}

export function constant()
{
    const config = { value: 0.1 };
    return tf.initializers.constant(config);
}
