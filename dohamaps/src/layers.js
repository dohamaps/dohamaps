
import * as tf from "@tensorflow/tfjs-node";

import * as initializers from "./initializers";

export function conv2d(options)
{
    const config =
    {
        filters: options.filters,
        kernelSize: options.kernelSize,
        strides: 1,
        padding: options.padding,
        dataFormat: "channelsLast",
        activation: options.activation ? options.activation : "relu",
        useBias: true,
        kernelInitializer: initializers.truncatedNormal(),
        biasInitializer: initializers.constant(),
    };
    return tf.layers.conv2d(config);
}

export function maxPooling2d(options)
{
    const config =
    {
        poolSize: options.poolSize ? options.poolSize : 2,
        strides: 2,
        padding: "valid",
        dataFormat: "channelsLast",
    };
    return tf.layers.maxPooling2d(config);
}

export function flatten(options)
{
    const config = { dataFormat: "channelsLast" };
    return tf.layers.flatten(config);
}

export function dense(options)
{
    const config =
    {
        units: options.units,
        activation: options.activation ? options.activation : "relu",
        useBias: true,
        kernelInitializer: initializers.truncatedNormal(),
        biasInitializer: initializers.constant(),
    }
    return tf.layers.dense(config);
}

export function upSampling2d(options)
{
    const config =
    {
        size: options.size ? options.size : 2,
        dataFormat: "channelsLast",
    }
    return tf.layers.upSampling2d(config);
}

class Block extends tf.layers.Layer
{
    constructor(layerList)
    {
        super.constructor();
        this.layerList = layerList;
    }
    apply(inputs)
    {
        super.apply(inputs);
        var outputs = inputs;
        for (let layer of this.layerList)
            outputs = layer.apply(outputs);
        return outputs;
    }
    computeOutputShape(inputShape)
    {
        outputShape = inputShape;
        for (let layer of this.layerList)
            outputShape = layer.computeOutputShape(outputShape);
        return outputShape;
    }
};

export function block(layerList)
{
    return new Block(layerList);
}
